//! distil-proxy — Context optimization server for LLM agents.
//!
//! Two modes of operation:
//!
//! 1. **Proxy mode** — sits between any LLM client and the real API. Intercepts
//!    chat/completions requests, optimizes context, forwards upstream.
//!    Zero code changes in the client — just point its base URL here.
//!
//! 2. **Direct API mode** — standalone optimization server. Clients POST
//!    messages/tools to `/v1/optimize` and get back optimized context + metrics.
//!    Works with any language or framework.
//!
//! # Usage
//!
//! ```bash
//! # Proxy mode — forward to Anthropic
//! distil-proxy --upstream https://api.anthropic.com/v1 --port 8080
//!
//! # Direct API mode — no upstream needed
//! distil-proxy --port 8080
//!
//! # Then:
//! # curl -X POST http://localhost:8080/v1/optimize -d '{"messages":[...], "tools":[...]}'
//! ```
//!
//! # Environment variables
//! - `DISTIL_UPSTREAM` — upstream API base URL (overrides --upstream)
//! - `DISTIL_PORT`     — listen port (overrides --port, default 8080)
//! - `DISTIL_BUDGET`   — token budget (overrides --budget, default 32000)
//! - `DISTIL_MODEL`    — model name for token counting (e.g. "gpt-4o")
//! - `DISTIL_SUMMARIZER_ENDPOINT` — LLM endpoint for SummarizationLayer
//! - `DISTIL_SUMMARIZER_MODEL`    — model for summarization (e.g. "claude-haiku-4-5-20251001")
//! - `DISTIL_SUMMARIZER_API_KEY`  — API key for the summarizer endpoint

use axum::{
    Router,
    body::Body,
    extract::{Request, State},
    http::{HeaderMap, HeaderName, HeaderValue, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Json, Response},
    routing::{get, post},
};
use distil::Layer;
use futures_util::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::sync::Arc;
use tower_http::{cors::CorsLayer, trace::TraceLayer};

// ── CLI args ─────────────────────────────────────────────────────────────────

#[derive(Debug)]
struct Config {
    upstream: Option<String>,
    port: u16,
    budget: usize,
    model: String,
    summarizer_endpoint: Option<String>,
    summarizer_model: Option<String>,
    summarizer_api_key: Option<String>,
    /// Path to distil.toml for declarative pipeline configuration.
    pipeline_config: Option<distil::PipelineConfig>,
    /// Bearer token for authenticating requests to distil's own endpoints.
    /// When set, all requests must include `Authorization: Bearer <token>`.
    /// Does NOT apply to upstream forwarding headers — those are passed through as-is.
    api_key: Option<String>,
}

impl Config {
    fn build_summarizer(&self) -> Option<distil::HttpSummarizer> {
        let endpoint = self.summarizer_endpoint.as_ref()?;
        let model = self.summarizer_model.as_ref()?;
        let api_key = self.summarizer_api_key.as_ref()?;
        Some(distil::HttpSummarizer::new(endpoint, model, api_key))
    }

    fn from_env_and_args() -> Self {
        let args: Vec<String> = std::env::args().collect();
        let get_flag = |flag: &str| -> Option<String> {
            args.windows(2).find(|w| w[0] == flag).map(|w| w[1].clone())
        };

        let upstream = std::env::var("DISTIL_UPSTREAM")
            .or_else(|_| get_flag("--upstream").ok_or(()))
            .ok();

        // Try loading a TOML pipeline config
        let config_path = std::env::var("DISTIL_CONFIG")
            .or_else(|_| get_flag("--config").ok_or(()))
            .ok();

        let pipeline_config =
            config_path.and_then(|path| match distil::PipelineConfig::from_file(&path) {
                Ok(cfg) => Some(cfg),
                Err(e) => {
                    eprintln!("warning: failed to load config from {path}: {e}");
                    None
                }
            });

        Self {
            upstream,
            port: std::env::var("DISTIL_PORT")
                .or_else(|_| get_flag("--port").ok_or(()))
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(8080),
            budget: std::env::var("DISTIL_BUDGET")
                .or_else(|_| get_flag("--budget").ok_or(()))
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(32_000),
            model: std::env::var("DISTIL_MODEL")
                .or_else(|_| get_flag("--model").ok_or(()))
                .unwrap_or_else(|_| "gpt-4o".into()),
            summarizer_endpoint: std::env::var("DISTIL_SUMMARIZER_ENDPOINT").ok(),
            summarizer_model: std::env::var("DISTIL_SUMMARIZER_MODEL").ok(),
            summarizer_api_key: std::env::var("DISTIL_SUMMARIZER_API_KEY").ok(),
            api_key: std::env::var("DISTIL_API_KEY")
                .or_else(|_| get_flag("--api-key").ok_or(()))
                .ok(),
            pipeline_config,
        }
    }
}

// ── Shared state ──────────────────────────────────────────────────────────────

struct ProxyState {
    config: Config,
    client: Client,
    scratchpad: distil::ScratchpadLayer,
    #[cfg(feature = "metrics")]
    metrics: distil::DistilMetrics,
}

// ── OpenAI wire types (subset we care about) ──────────────────────────────────

#[derive(Debug, Clone, Deserialize, Serialize)]
struct OaiMessage {
    role: String,
    #[serde(default, skip_serializing_if = "Value::is_null")]
    content: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Value>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct OaiTool {
    #[serde(rename = "type")]
    kind: String,
    function: OaiFunctionDef,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct OaiFunctionDef {
    name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    parameters: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct ChatRequest {
    model: Option<String>,
    messages: Vec<OaiMessage>,
    #[serde(default)]
    tools: Vec<OaiTool>,
    #[serde(flatten)]
    extra: Value,
}

// ── Direct API types ─────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct OptimizeRequest {
    messages: Vec<OaiMessage>,
    #[serde(default)]
    tools: Vec<OaiTool>,
    #[serde(default)]
    turn: Option<u32>,
    #[serde(default)]
    config: Option<OptimizeConfig>,
}

#[derive(Debug, Deserialize)]
struct OptimizeConfig {
    budget: Option<usize>,
    retain_turns: Option<u32>,
    retain_turns_tool: Option<u32>,
    retain_turns_assistant: Option<u32>,
    preserve_recent: Option<usize>,
    json_truncate: Option<bool>,
}

#[derive(Debug, Serialize)]
struct OptimizeResponse {
    messages: Vec<OaiMessage>,
    tools: Vec<OaiTool>,
    metrics: OptimizeMetrics,
}

#[derive(Debug, Serialize)]
struct OptimizeMetrics {
    tokens_before: usize,
    tokens_after: usize,
    tokens_saved: usize,
    percentage_saved: f64,
    duration_ms: f64,
    layers: Vec<LayerMetric>,
}

#[derive(Debug, Serialize)]
struct LayerMetric {
    name: String,
    tokens_before: usize,
    tokens_after: usize,
    tokens_saved: usize,
    duration_ms: f64,
    detail: String,
}

#[derive(Debug, Deserialize)]
struct ToolCallRequest {
    name: String,
    arguments: Value,
    #[serde(default)]
    tools: Vec<OaiTool>,
}

#[derive(Debug, Serialize)]
struct ToolCallResponse {
    output: Option<String>,
    error: Option<String>,
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn oai_tools_to_specs(tools: &[OaiTool]) -> Vec<distil::ToolSpec> {
    tools
        .iter()
        .filter(|t| t.kind == "function")
        .map(|t| distil::ToolSpec {
            name: t.function.name.clone(),
            description: t.function.description.clone().unwrap_or_default(),
            parameters: t
                .function
                .parameters
                .clone()
                .unwrap_or_else(|| json!({"type": "object", "properties": {}})),
        })
        .collect()
}

fn specs_to_oai_tools(specs: &[distil::ToolSpec]) -> Vec<OaiTool> {
    specs
        .iter()
        .map(|t| OaiTool {
            kind: "function".into(),
            function: OaiFunctionDef {
                name: t.name.clone(),
                description: Some(t.description.clone()),
                parameters: Some(t.parameters.clone()),
            },
        })
        .collect()
}

fn to_distil_messages(msgs: &[OaiMessage]) -> Vec<distil::Message> {
    msgs.iter()
        .map(|m| {
            let content = match &m.content {
                Value::String(s) => s.clone(),
                Value::Null => String::new(),
                other => other.to_string(),
            };
            let role = match m.role.as_str() {
                "assistant" => distil::types::Role::Assistant,
                "system" => distil::types::Role::System,
                "tool" => distil::types::Role::Tool,
                _ => distil::types::Role::User,
            };
            distil::Message { role, content }
        })
        .collect()
}

fn to_oai_messages(msgs: Vec<distil::Message>, originals: &[OaiMessage]) -> Vec<OaiMessage> {
    let orig_len = originals.len();
    let new_len = msgs.len();
    let offset = orig_len.saturating_sub(new_len);

    msgs.into_iter()
        .enumerate()
        .map(|(i, m)| {
            let content_str = m.content;
            let orig_idx = i + offset;
            if let Some(orig) = originals.get(orig_idx) {
                OaiMessage {
                    role: role_str(&m.role).to_string(),
                    content: Value::String(content_str),
                    name: orig.name.clone(),
                    tool_call_id: orig.tool_call_id.clone(),
                    tool_calls: orig.tool_calls.clone(),
                }
            } else {
                OaiMessage {
                    role: role_str(&m.role).to_string(),
                    content: Value::String(content_str),
                    name: None,
                    tool_call_id: None,
                    tool_calls: None,
                }
            }
        })
        .collect()
}

fn role_str(role: &distil::types::Role) -> &'static str {
    match role {
        distil::types::Role::User => "user",
        distil::types::Role::Assistant => "assistant",
        distil::types::Role::System => "system",
        distil::types::Role::Tool => "tool",
    }
}

/// RFC 9457-compliant structured error response.
/// Delegates to `distil::http::error_body()` for the JSON structure.
fn error_response(status: StatusCode, message: &str) -> (StatusCode, Json<Value>) {
    let body = distil::http::error_body(status.as_u16(), message);
    (status, Json(body))
}

fn build_pipeline_from_config(
    tools: &[OaiTool],
    config: &OptimizeConfig,
    default_budget: usize,
    summarizer: Option<distil::HttpSummarizer>,
) -> distil::Pipeline {
    let counter = distil::EstimateCounter;
    let tool_specs = oai_tools_to_specs(tools);

    let mut builder = distil::Pipeline::builder().counter(distil::EstimateCounter);

    if !tool_specs.is_empty() {
        builder = builder.layer(distil::RegistryLayer::new(tool_specs, &counter));
    }

    let mut masking = distil::MaskingLayer::new();
    if let Some(rt) = config.retain_turns {
        masking = masking.retain_turns(rt);
    }
    if let Some(rt) = config.retain_turns_tool {
        masking = masking.retain_turns_tool(rt);
    }
    if let Some(ra) = config.retain_turns_assistant {
        masking = masking.retain_turns_assistant(ra);
    }
    if config.json_truncate == Some(false) {
        masking = masking.no_json_truncate();
    }
    builder = builder.layer(masking);

    if let Some(s) = summarizer {
        builder = builder.layer(distil::SummarizationLayer::new(s));
    }

    builder = builder.layer(distil::CompactionLayer::new());

    let budget = config.budget.unwrap_or(default_budget);
    let mut budget_layer = distil::BudgetLayer::new(budget);
    if let Some(pr) = config.preserve_recent {
        budget_layer = budget_layer.preserve_recent(pr);
    }
    builder = builder.layer(budget_layer);

    builder.build()
}

fn build_default_pipeline(
    tools: &[OaiTool],
    budget: usize,
    summarizer: Option<distil::HttpSummarizer>,
) -> distil::Pipeline {
    let counter = distil::EstimateCounter;
    let tool_specs = oai_tools_to_specs(tools);

    let mut builder = distil::Pipeline::builder().counter(distil::EstimateCounter);

    if !tool_specs.is_empty() {
        builder = builder.layer(distil::RegistryLayer::new(tool_specs, &counter));
    }

    builder = builder.layer(distil::MaskingLayer::new().retain_turns(3));

    if let Some(s) = summarizer {
        builder = builder.layer(distil::SummarizationLayer::new(s));
    }

    builder = builder
        .layer(distil::CompactionLayer::new())
        .layer(distil::BudgetLayer::new(budget));

    builder.build()
}

fn pipeline_result_to_metrics(result: &distil::PipelineResult) -> OptimizeMetrics {
    OptimizeMetrics {
        tokens_before: result.tokens_before,
        tokens_after: result.tokens_after,
        tokens_saved: result.total_saved(),
        percentage_saved: result.percentage_saved(),
        duration_ms: result.duration.as_secs_f64() * 1000.0,
        layers: result
            .layers
            .iter()
            .map(|lr| LayerMetric {
                name: lr.layer.clone(),
                tokens_before: lr.tokens_before,
                tokens_after: lr.tokens_after,
                tokens_saved: lr.tokens_saved(),
                duration_ms: lr.duration.as_secs_f64() * 1000.0,
                detail: lr.detail.clone(),
            })
            .collect(),
    }
}

// ── POST /v1/optimize — Direct optimization API ─────────────────────────────

async fn optimize(
    State(state): State<Arc<ProxyState>>,
    Json(body): Json<OptimizeRequest>,
) -> Result<Json<OptimizeResponse>, (StatusCode, Json<Value>)> {
    let distil_messages = to_distil_messages(&body.messages);
    let tool_specs = oai_tools_to_specs(&body.tools);

    let turn = body
        .turn
        .unwrap_or_else(|| body.messages.iter().filter(|m| m.role == "user").count() as u32);

    let pipeline = if let Some(ref pipeline_config) = state.config.pipeline_config {
        // TOML-configured pipeline takes precedence
        pipeline_config.build_pipeline(
            &oai_tools_to_specs(&body.tools),
            state.config.build_summarizer(),
            None,
        )
    } else if let Some(ref config) = body.config {
        build_pipeline_from_config(
            &body.tools,
            config,
            state.config.budget,
            state.config.build_summarizer(),
        )
    } else {
        build_default_pipeline(
            &body.tools,
            state.config.budget,
            state.config.build_summarizer(),
        )
    };

    let mut ctx = distil::Ctx::new(distil_messages, tool_specs, turn);
    let result = pipeline.optimize(&mut ctx);

    tracing::info!(
        tokens_before = result.tokens_before,
        tokens_after = result.tokens_after,
        saved = result.total_saved(),
        "optimize: {:.1}% saved",
        result.percentage_saved()
    );

    #[cfg(feature = "metrics")]
    state.metrics.record(&result);

    let optimized_messages = to_oai_messages(ctx.messages, &body.messages);
    let optimized_tools = specs_to_oai_tools(&ctx.tools);

    Ok(Json(OptimizeResponse {
        messages: optimized_messages,
        tools: optimized_tools,
        metrics: pipeline_result_to_metrics(&result),
    }))
}

// ── POST /v1/tool_call — Handle distil-injected tool calls ──────────────────

async fn tool_call(
    State(state): State<Arc<ProxyState>>,
    Json(body): Json<ToolCallRequest>,
) -> Json<ToolCallResponse> {
    // Try scratchpad tools first (note_write, note_read)
    if let Some(output) = state
        .scratchpad
        .handle_tool_call(&body.name, &body.arguments)
    {
        return Json(ToolCallResponse {
            output: Some(output),
            error: None,
        });
    }

    // Try tool_search — needs a RegistryLayer built from provided tools
    if body.name == "tool_search" && !body.tools.is_empty() {
        let counter = distil::EstimateCounter;
        let tool_specs = oai_tools_to_specs(&body.tools);
        let registry = distil::RegistryLayer::new(tool_specs, &counter);
        if let Some(output) = registry.handle_tool_call(&body.name, &body.arguments) {
            return Json(ToolCallResponse {
                output: Some(output),
                error: None,
            });
        }
    }

    Json(ToolCallResponse {
        output: None,
        error: Some(format!("unknown tool: {}", body.name)),
    })
}

// ── Proxy: POST /v1/chat/completions ─────────────────────────────────────────

async fn chat_completions(
    State(state): State<Arc<ProxyState>>,
    headers: HeaderMap,
    Json(body): Json<ChatRequest>,
) -> Response {
    let upstream = match &state.config.upstream {
        Some(u) => u,
        None => {
            return error_response(
                StatusCode::BAD_REQUEST,
                "no upstream configured — use /v1/optimize for direct API mode, or set DISTIL_UPSTREAM",
            )
            .into_response();
        }
    };

    let model = body
        .model
        .clone()
        .unwrap_or_else(|| state.config.model.clone());

    let tool_specs = oai_tools_to_specs(&body.tools);

    let pipeline = if let Some(ref pipeline_config) = state.config.pipeline_config {
        pipeline_config.build_pipeline(&tool_specs, state.config.build_summarizer(), None)
    } else {
        build_default_pipeline(
            &body.tools,
            state.config.budget,
            state.config.build_summarizer(),
        )
    };

    let distil_messages = to_distil_messages(&body.messages);
    let turn = body.messages.iter().filter(|m| m.role == "user").count();

    let mut ctx = distil::Ctx::new(distil_messages, tool_specs, turn as u32);
    let result = pipeline.optimize(&mut ctx);

    let tokens_before = result.tokens_before;
    let tokens_after = result.tokens_after;
    let saved = result.total_saved();

    tracing::info!(
        model = %model,
        tokens_before,
        tokens_after,
        saved,
        "distil: {:.1}% saved",
        result.percentage_saved()
    );

    #[cfg(feature = "metrics")]
    state.metrics.record(&result);

    let optimized_messages = to_oai_messages(ctx.messages, &body.messages);

    // Build the upstream payload
    let mut upstream_body = body.extra.clone();
    if !upstream_body.is_object() {
        upstream_body = json!({});
    }
    let obj = upstream_body.as_object_mut().unwrap();
    obj.insert("messages".into(), json!(optimized_messages));
    if let Some(m) = &body.model {
        obj.insert("model".into(), json!(m));
    }
    if !body.tools.is_empty() && ctx.tools.len() == body.tools.len() {
        obj.insert("tools".into(), json!(body.tools));
    } else if !ctx.tools.is_empty() {
        obj.insert("tools".into(), json!(specs_to_oai_tools(&ctx.tools)));
    }

    if let Some(catalog) = &ctx.catalog {
        if let Some(msgs) = obj.get_mut("messages").and_then(|m| m.as_array_mut()) {
            let catalog_msg = json!({
                "role": "system",
                "content": format!("Available tools (call tool_search to get full schemas):\n\n{catalog}")
            });
            msgs.insert(0, catalog_msg);
        }
    }

    let upstream_url = format!("{upstream}/chat/completions");

    let mut req = state.client.post(&upstream_url).json(&upstream_body);

    for (k, v) in &headers {
        let name = k.as_str().to_lowercase();
        if name == "authorization"
            || name == "x-api-key"
            || name == "anthropic-version"
            || name == "content-type"
        {
            req = req.header(k, v);
        }
    }

    let upstream_resp = match req.send().await {
        Ok(r) => r,
        Err(e) => {
            return error_response(StatusCode::BAD_GATEWAY, &format!("upstream error: {e}"))
                .into_response();
        }
    };

    let status = upstream_resp.status();
    let resp_headers = upstream_resp.headers().clone();

    // Check if upstream is streaming (SSE)
    let is_streaming = resp_headers
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .is_some_and(|ct| ct.contains("text/event-stream"));

    let mut distil_headers = HeaderMap::new();
    for (k, v) in &resp_headers {
        let name = k.as_str();
        if name == "content-type" || name == "transfer-encoding" || name == "cache-control" {
            let _ = distil_headers.insert(k, v.clone());
        }
    }
    let _ = distil_headers.insert(
        HeaderName::from_static("x-distil-tokens-before"),
        HeaderValue::from_str(&tokens_before.to_string()).unwrap(),
    );
    let _ = distil_headers.insert(
        HeaderName::from_static("x-distil-tokens-after"),
        HeaderValue::from_str(&tokens_after.to_string()).unwrap(),
    );
    let _ = distil_headers.insert(
        HeaderName::from_static("x-distil-tokens-saved"),
        HeaderValue::from_str(&saved.to_string()).unwrap(),
    );

    if is_streaming {
        // Stream SSE chunks through without buffering
        let byte_stream = upstream_resp
            .bytes_stream()
            .map(|chunk| chunk.map_err(axum::Error::new));
        let body = Body::from_stream(byte_stream);

        let mut response = Response::builder().status(status).body(body).unwrap();
        *response.headers_mut() = distil_headers;
        response
    } else {
        // Non-streaming: buffer the full response
        let resp_bytes = match upstream_resp.bytes().await {
            Ok(b) => b,
            Err(e) => {
                return error_response(
                    StatusCode::BAD_GATEWAY,
                    &format!("upstream read error: {e}"),
                )
                .into_response();
            }
        };

        let mut response = Response::builder()
            .status(status)
            .body(Body::from(resp_bytes))
            .unwrap();
        *response.headers_mut() = distil_headers;
        response
    }
}

// ── GET /v1/health ───────────────────────────────────────────────────────────

async fn health(State(state): State<Arc<ProxyState>>) -> Json<Value> {
    Json(json!({
        "ok": true,
        "service": "distil-proxy",
        "version": env!("CARGO_PKG_VERSION"),
        "mode": if state.config.upstream.is_some() { "proxy" } else { "direct" },
        "budget": state.config.budget,
        "summarizer": state.config.summarizer_endpoint.is_some(),
        "metrics": cfg!(feature = "metrics"),
    }))
}

// ── GET /metrics — Prometheus metrics endpoint ───────────────────────────────

#[cfg(feature = "metrics")]
async fn prometheus_metrics(State(state): State<Arc<ProxyState>>) -> impl IntoResponse {
    let body = state.metrics.render();
    (
        StatusCode::OK,
        [(
            HeaderName::from_static("content-type"),
            HeaderValue::from_static("text/plain; version=0.0.4; charset=utf-8"),
        )],
        body,
    )
}

// ── Auth middleware ───────────────────────────────────────────────────────────

/// Bearer token authentication middleware.
/// Skips auth for health endpoints. Returns RFC 9457 error on failure.
async fn auth_middleware(
    State(state): State<Arc<ProxyState>>,
    req: Request,
    next: Next,
) -> Response {
    let auth_header = req
        .headers()
        .get("authorization")
        .and_then(|v| v.to_str().ok());
    let path = req.uri().path();

    if distil::http::check_bearer_auth(auth_header, path, state.config.api_key.as_deref()) {
        next.run(req).await
    } else {
        error_response(
            StatusCode::UNAUTHORIZED,
            "invalid or missing Authorization: Bearer <token>",
        )
        .into_response()
    }
}

// ── Main ──────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            std::env::var("RUST_LOG")
                .unwrap_or_else(|_| "distil_proxy=info,tower_http=warn".into()),
        )
        .init();

    let config = Config::from_env_and_args();
    let port = config.port;
    let mode = if config.upstream.is_some() {
        "proxy"
    } else {
        "direct"
    };

    tracing::info!(
        upstream = config.upstream.as_deref().unwrap_or("none"),
        port,
        budget = config.budget,
        mode,
        auth = config.api_key.is_some(),
        "distil-proxy starting"
    );

    let state = Arc::new(ProxyState {
        config,
        client: Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .build()
            .expect("failed to build HTTP client"),
        scratchpad: distil::ScratchpadLayer::new(),
        #[cfg(feature = "metrics")]
        metrics: distil::DistilMetrics::new(),
    });

    let app = Router::new()
        // Direct API endpoints
        .route("/v1/optimize", post(optimize))
        .route("/v1/tool_call", post(tool_call))
        .route("/v1/health", get(health))
        .route("/health", get(health))
        // Proxy endpoints
        .route("/v1/chat/completions", post(chat_completions))
        .route("/chat/completions", post(chat_completions));

    // Metrics endpoint (only when feature enabled)
    #[cfg(feature = "metrics")]
    let app = {
        tracing::info!("prometheus metrics enabled at /metrics");
        app.route("/metrics", get(prometheus_metrics))
            .route("/v1/metrics", get(prometheus_metrics))
    };

    let app = app
        .layer(middleware::from_fn_with_state(
            state.clone(),
            auth_middleware,
        ))
        .with_state(state)
        .layer(TraceLayer::new_for_http())
        .layer(CorsLayer::permissive());

    let addr = format!("0.0.0.0:{port}");
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .unwrap_or_else(|e| panic!("failed to bind {addr}: {e}"));

    tracing::info!("listening on {addr}");
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .expect("server failed");
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        () = ctrl_c => tracing::info!("received Ctrl+C, shutting down"),
        () = terminate => tracing::info!("received SIGTERM, shutting down"),
    }
}
