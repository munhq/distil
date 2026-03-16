//! distil-mcp — MCP server for context optimization.
//!
//! Exposes distil's pipeline as an MCP (Model Context Protocol) server.
//! Any MCP-compatible client (Claude Code, Cursor, VS Code Copilot) can use
//! distil natively via stdio transport.
//!
//! # Usage
//!
//! ```bash
//! distil-mcp
//! ```
//!
//! # MCP Configuration (claude_desktop_config.json)
//!
//! ```json
//! {
//!   "mcpServers": {
//!     "distil": {
//!       "command": "distil-mcp"
//!     }
//!   }
//! }
//! ```

use std::sync::Arc;

use rmcp::{
    handler::server::{tool::ToolRouter, wrapper::Parameters},
    model::{CallToolResult, Content, Implementation, ServerCapabilities},
    tool, tool_router, ErrorData as McpError, ServerHandler,
    service::ServiceExt,
    transport::io::stdio,
};
use schemars::JsonSchema;
use serde::Deserialize;

use distil::{
    counter::EstimateCounter,
    layers::*,
    Layer,
    pipeline::{Ctx, Pipeline},
    types::{Message, Role, ToolSpec},
};

// ── MCP Request Types ─────────────────────────────────────────────────────────

#[derive(Debug, Deserialize, JsonSchema)]
struct OptimizeRequest {
    /// Conversation messages in [{role, content}] format.
    messages: Vec<McpMessage>,
    /// Tool definitions in [{name, description, parameters}] format.
    #[serde(default)]
    tools: Vec<McpTool>,
    /// Current conversation turn (auto-detected if omitted).
    #[serde(default)]
    turn: Option<u32>,
    /// Token budget (default: 32000).
    #[serde(default = "default_budget")]
    budget: usize,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct McpMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct McpTool {
    name: String,
    #[serde(default)]
    description: String,
    #[serde(default)]
    parameters: serde_json::Value,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct ToolCallRequest {
    /// Tool name to invoke (e.g., "tool_search", "note_read", "note_write").
    name: String,
    /// Tool arguments as a JSON object.
    #[serde(default)]
    arguments: serde_json::Value,
    /// Tool definitions (needed for tool_search).
    #[serde(default)]
    tools: Vec<McpTool>,
}

fn default_budget() -> usize {
    32_000
}

// ── Conversion helpers ─────────────────────────────────────────────────────────

fn to_distil_messages(msgs: &[McpMessage]) -> Vec<Message> {
    msgs.iter()
        .map(|m| {
            let role = match m.role.as_str() {
                "assistant" => Role::Assistant,
                "system" => Role::System,
                "tool" => Role::Tool,
                _ => Role::User,
            };
            Message { role, content: m.content.clone() }
        })
        .collect()
}

fn to_distil_tools(tools: &[McpTool]) -> Vec<ToolSpec> {
    tools
        .iter()
        .map(|t| ToolSpec {
            name: t.name.clone(),
            description: t.description.clone(),
            parameters: if t.parameters.is_null() {
                serde_json::json!({"type": "object", "properties": {}})
            } else {
                t.parameters.clone()
            },
        })
        .collect()
}

// ── MCP Server ────────────────────────────────────────────────────────────────

#[derive(Clone)]
#[allow(dead_code)]
struct DistilServer {
    scratchpad: Arc<ScratchpadLayer>,
    tool_router: ToolRouter<Self>,
}

#[tool_router]
impl DistilServer {
    fn new() -> Self {
        Self {
            scratchpad: Arc::new(ScratchpadLayer::new()),
            tool_router: Self::tool_router(),
        }
    }

    #[tool(description = "Optimize conversation context for LLM token efficiency. Takes messages and tool definitions, returns optimized context with metrics. Typically saves 50-90% of tokens.")]
    async fn optimize(
        &self,
        Parameters(request): Parameters<OptimizeRequest>,
    ) -> Result<CallToolResult, McpError> {
        let counter = EstimateCounter;
        let messages = to_distil_messages(&request.messages);
        let tool_specs = to_distil_tools(&request.tools);

        let turn = request.turn.unwrap_or_else(|| {
            messages.iter().filter(|m| m.role == Role::User).count() as u32
        });

        // Build pipeline
        let mut builder = Pipeline::builder().counter(EstimateCounter);

        if !tool_specs.is_empty() {
            builder = builder.layer(RegistryLayer::new(tool_specs.clone(), &counter));
        }

        builder = builder.layer(MaskingLayer::new().retain_turns(3));
        builder = builder.layer(CompactionLayer::new());
        builder = builder.layer(BudgetLayer::new(request.budget).preserve_recent(6));
        builder = builder.layer(CacheAlignLayer::generic());

        let pipeline = builder.build();
        let mut ctx = Ctx::new(messages, tool_specs, turn);
        let result = pipeline.optimize(&mut ctx);

        // Format response
        let optimized_messages: Vec<serde_json::Value> = ctx
            .messages
            .iter()
            .map(|m| {
                serde_json::json!({
                    "role": match m.role {
                        Role::System => "system",
                        Role::User => "user",
                        Role::Assistant => "assistant",
                        Role::Tool => "tool",
                    },
                    "content": m.content,
                })
            })
            .collect();

        let optimized_tools: Vec<serde_json::Value> = ctx
            .tools
            .iter()
            .map(|t| {
                serde_json::json!({
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                })
            })
            .collect();

        let response = serde_json::json!({
            "messages": optimized_messages,
            "tools": optimized_tools,
            "metrics": {
                "tokens_before": result.tokens_before,
                "tokens_after": result.tokens_after,
                "tokens_saved": result.total_saved(),
                "percentage_saved": result.percentage_saved(),
                "duration_ms": result.duration.as_secs_f64() * 1000.0,
                "layers": result.layers.iter().map(|lr| serde_json::json!({
                    "name": lr.layer,
                    "tokens_before": lr.tokens_before,
                    "tokens_after": lr.tokens_after,
                    "tokens_saved": lr.tokens_saved(),
                    "duration_ms": lr.duration.as_secs_f64() * 1000.0,
                    "detail": lr.detail,
                })).collect::<Vec<_>>(),
            }
        });

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&response).unwrap_or_default(),
        )]))
    }

    #[tool(description = "Handle a distil-injected tool call (tool_search, note_read, note_write). When distil optimizes your tools, it injects meta-tools. Route those calls here.")]
    async fn tool_call(
        &self,
        Parameters(request): Parameters<ToolCallRequest>,
    ) -> Result<CallToolResult, McpError> {
        // Try scratchpad first
        if let Some(output) =
            Layer::handle_tool_call(&*self.scratchpad, &request.name, &request.arguments)
        {
            return Ok(CallToolResult::success(vec![Content::text(output)]));
        }

        // Try tool_search
        if request.name == "tool_search" && !request.tools.is_empty() {
            let counter = EstimateCounter;
            let tool_specs = to_distil_tools(&request.tools);
            let registry = RegistryLayer::new(tool_specs, &counter);
            if let Some(output) = Layer::handle_tool_call(&registry, &request.name, &request.arguments) {
                return Ok(CallToolResult::success(vec![Content::text(output)]));
            }
        }

        Err(McpError::invalid_params(
            format!("unknown distil tool: {}", request.name),
            None,
        ))
    }
}

// ── Implement ServerHandler ───────────────────────────────────────────────────

impl ServerHandler for DistilServer {
    fn get_info(&self) -> rmcp::model::ServerInfo {
        rmcp::model::ServerInfo {
            protocol_version: Default::default(),
            capabilities: ServerCapabilities::builder()
                .enable_tools()
                .build(),
            server_info: Implementation {
                name: "distil".into(),
                version: env!("CARGO_PKG_VERSION").into(),
                title: Some("Distil Context Optimizer".into()),
                description: Some("Context optimization middleware for LLM agents — 50-90% token savings".into()),
                icons: None,
                website_url: None,
            },
            instructions: Some(
                "Use 'optimize' to compress conversation context before sending to an LLM. \
                 Use 'tool_call' to handle distil-injected meta-tools (tool_search, note_read, note_write)."
                    .into(),
            ),
        }
    }
}

// ── Main ──────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    // MCP servers must not write to stdout (it's the JSON-RPC channel)
    eprintln!("distil-mcp v{} starting (stdio transport)", env!("CARGO_PKG_VERSION"));

    let server = DistilServer::new();
    let service = match server.serve(stdio()).await {
        Ok(s) => s,
        Err(e) => {
            eprintln!("failed to start MCP service: {e}");
            std::process::exit(1);
        }
    };

    if let Err(e) = service.waiting().await {
        eprintln!("MCP service error: {e}");
        std::process::exit(1);
    }
}
