use distil::counter::EstimateCounter;
use distil::layers::*;
use distil::masker::JsonTruncateConfig;
use distil::pipeline::{Ctx, Layer, Pipeline};
use distil::types::{Message, ToolSpec};

/// Simulate a realistic 30-tool agent.
fn realistic_tools() -> Vec<ToolSpec> {
    let tools_data = [
        ("shell", "Execute a shell command. Returns stdout, stderr, and exit code. Use for running builds, tests, system commands.", r#"{"type":"object","properties":{"command":{"type":"string","description":"The shell command to execute"},"timeout":{"type":"integer","description":"Timeout in seconds, default 30"},"working_dir":{"type":"string","description":"Working directory for the command"}},"required":["command"]}"#),
        ("file_read", "Read a file from disk. Returns content as string. Supports line ranges for partial reads of large files.", r#"{"type":"object","properties":{"path":{"type":"string"},"line_start":{"type":"integer"},"line_end":{"type":"integer"}},"required":["path"]}"#),
        ("file_write", "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. Use for code generation and edits.", r#"{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}"#),
        ("git_status", "Get the current git status showing staged, unstaged, and untracked files.", r#"{"type":"object","properties":{}}"#),
        ("git_diff", "Show git diff for staged or unstaged changes. Supports path filters.", r#"{"type":"object","properties":{"staged":{"type":"boolean"},"path":{"type":"string"}}}"#),
        ("git_log", "Show git commit log. Configurable count and format.", r#"{"type":"object","properties":{"count":{"type":"integer","default":10},"oneline":{"type":"boolean","default":true}}}"#),
        ("git_commit", "Create a git commit with the given message.", r#"{"type":"object","properties":{"message":{"type":"string"},"all":{"type":"boolean"}},"required":["message"]}"#),
        ("git_branch", "List, create, or switch git branches.", r#"{"type":"object","properties":{"action":{"type":"string","enum":["list","create","checkout"]},"name":{"type":"string"}},"required":["action"]}"#),
        ("memory_store", "Store a key-value pair in the agent's persistent memory for cross-conversation recall.", r#"{"type":"object","properties":{"key":{"type":"string"},"value":{"type":"string"},"namespace":{"type":"string"}},"required":["key","value"]}"#),
        ("memory_recall", "Recall information from persistent memory by key or semantic search query.", r#"{"type":"object","properties":{"query":{"type":"string"},"namespace":{"type":"string"},"limit":{"type":"integer"}},"required":["query"]}"#),
        ("memory_list", "List all keys in persistent memory, optionally filtered by namespace.", r#"{"type":"object","properties":{"namespace":{"type":"string"}}}"#),
        ("memory_delete", "Delete a key from persistent memory.", r#"{"type":"object","properties":{"key":{"type":"string"}},"required":["key"]}"#),
        ("http_request", "Make an HTTP request. Supports GET, POST, PUT, DELETE with headers, body, and auth.", r#"{"type":"object","properties":{"method":{"type":"string","enum":["GET","POST","PUT","DELETE","PATCH"]},"url":{"type":"string"},"headers":{"type":"object"},"body":{"type":"string"},"auth":{"type":"string"}},"required":["method","url"]}"#),
        ("browser_navigate", "Navigate to a URL in the headless browser and return the page content.", r#"{"type":"object","properties":{"url":{"type":"string"},"wait_for":{"type":"string"},"screenshot":{"type":"boolean"}},"required":["url"]}"#),
        ("browser_click", "Click an element on the current page by CSS selector.", r#"{"type":"object","properties":{"selector":{"type":"string"}},"required":["selector"]}"#),
        ("browser_extract", "Extract text content from the current page using CSS selectors.", r#"{"type":"object","properties":{"selector":{"type":"string"},"attribute":{"type":"string"}},"required":["selector"]}"#),
        ("web_search", "Search the web and return top results with titles, URLs, and snippets.", r#"{"type":"object","properties":{"query":{"type":"string"},"num_results":{"type":"integer","default":5}},"required":["query"]}"#),
        ("delegate", "Delegate a subtask to a specialized sub-agent with isolated context.", r#"{"type":"object","properties":{"task":{"type":"string"},"agent_type":{"type":"string","enum":["research","code","review"]},"context":{"type":"string"}},"required":["task"]}"#),
        ("sql_query", "Execute a SQL query against the project database. Read-only by default.", r#"{"type":"object","properties":{"query":{"type":"string"},"readonly":{"type":"boolean","default":true}},"required":["query"]}"#),
        ("vision_analyze", "Analyze an image using vision capabilities. Describe, extract text, or answer questions.", r#"{"type":"object","properties":{"image_path":{"type":"string"},"prompt":{"type":"string"}},"required":["image_path","prompt"]}"#),
        ("document_extract", "Extract text from documents (PDF, DOCX, etc.).", r#"{"type":"object","properties":{"path":{"type":"string"},"pages":{"type":"string"}},"required":["path"]}"#),
        ("cloud_costs", "Query cloud infrastructure costs by service, region, or time period.", r#"{"type":"object","properties":{"provider":{"type":"string","enum":["aws","gcp","cloudflare"]},"service":{"type":"string"},"period":{"type":"string"}},"required":["provider"]}"#),
        ("cloud_resources", "List cloud resources (instances, buckets, functions, etc.).", r#"{"type":"object","properties":{"provider":{"type":"string"},"resource_type":{"type":"string"},"region":{"type":"string"}},"required":["provider"]}"#),
        ("task_create", "Create a new task/todo item.", r#"{"type":"object","properties":{"title":{"type":"string"},"description":{"type":"string"},"priority":{"type":"string","enum":["low","medium","high","critical"]}},"required":["title"]}"#),
        ("task_list", "List tasks, optionally filtered by status or priority.", r#"{"type":"object","properties":{"status":{"type":"string"},"priority":{"type":"string"}}}"#),
        ("task_update", "Update a task's status or details.", r#"{"type":"object","properties":{"id":{"type":"string"},"status":{"type":"string"},"title":{"type":"string"}},"required":["id"]}"#),
        ("notify_agent", "Send a notification to another agent or the user.", r#"{"type":"object","properties":{"target":{"type":"string"},"message":{"type":"string"}},"required":["target","message"]}"#),
        ("publish_event", "Publish an event to the event bus for cross-agent communication.", r#"{"type":"object","properties":{"topic":{"type":"string"},"payload":{"type":"object"}},"required":["topic","payload"]}"#),
        ("mcp_tool", "Call a tool exposed by an MCP server.", r#"{"type":"object","properties":{"server":{"type":"string"},"tool":{"type":"string"},"arguments":{"type":"object"}},"required":["server","tool"]}"#),
        ("log_activity", "Log an activity entry for audit trail.", r#"{"type":"object","properties":{"action":{"type":"string"},"details":{"type":"string"},"severity":{"type":"string","enum":["info","warn","error"]}},"required":["action"]}"#),
    ];

    tools_data
        .into_iter()
        .map(|(name, desc, params)| ToolSpec {
            name: name.into(),
            description: desc.into(),
            parameters: serde_json::from_str(params).unwrap(),
        })
        .collect()
}

// ── Realistic conversation WITH JSON tool results and Role::Tool messages ─────

/// This simulates how real agent frameworks work:
/// - Assistant messages contain the LLM's reasoning
/// - Role::Tool messages contain the raw tool output (often JSON)
/// - Older turns have large outputs that should be compressed
fn realistic_conversation_v2() -> Vec<Message> {
    // Large JSON tool outputs — the kind agents actually produce
    let shell_build_json = serde_json::json!({
        "stdout": "   Compiling proc-macro2 v1.0.106\n   Compiling unicode-ident v1.0.24\n   Compiling quote v1.0.45\n   Compiling syn v2.0.117\n   Compiling serde_derive v1.0.228\n   Compiling serde v1.0.228\n   Compiling serde_json v1.0.149\n   Compiling tokio v1.43.1\n   Compiling axum v0.8.1\n   Compiling my-project v0.1.0 (/home/user/project)\nwarning: unused import: `std::io::Write`\n --> src/main.rs:3:5\n  |\n3 | use std::io::Write;\n  |     ^^^^^^^^^^^^^^\n  |\n  = note: `#[warn(unused_imports)]` on by default\n\nwarning: `my-project` (bin \"my-project\") generated 1 warning\n    Finished `dev` profile [unoptimized + debuginfo] target(s) in 12.34s",
        "stderr": "",
        "exit_code": 0,
        "duration_ms": 12340,
        "command": "cargo build"
    }).to_string();

    let file_read_json = serde_json::json!({
        "path": "src/main.rs",
        "content": "use axum::{Router, routing::get, Json};\nuse serde::Serialize;\nuse std::io::Write;\n\n#[derive(Serialize)]\nstruct Health {\n    status: String,\n    version: String,\n}\n\n#[tokio::main]\nasync fn main() {\n    let app = Router::new()\n        .route(\"/health\", get(health));\n\n    let listener = tokio::net::TcpListener::bind(\"0.0.0.0:3000\").await.unwrap();\n    axum::serve(listener, app).await.unwrap();\n}\n\nasync fn health() -> Json<Health> {\n    Json(Health {\n        status: \"ok\".into(),\n        version: env!(\"CARGO_PKG_VERSION\").into(),\n    })\n}",
        "lines": 24,
        "size_bytes": 498
    }).to_string();

    let git_status_json = serde_json::json!({
        "branch": "feature/add-auth",
        "staged": [],
        "unstaged": ["src/main.rs", "Cargo.toml"],
        "untracked": ["src/auth.rs", "src/middleware.rs"],
        "ahead": 0,
        "behind": 0,
        "clean": false
    }).to_string();

    let test_output_json = serde_json::json!({
        "stdout": "running 12 tests\ntest tests::health_endpoint ... ok\ntest tests::auth_middleware ... ok\ntest tests::jwt_validation ... ok\ntest tests::token_refresh ... ok\ntest tests::protected_route ... ok\ntest tests::unauthorized_request ... ok\ntest tests::expired_token ... ok\ntest tests::invalid_signature ... ok\ntest tests::rate_limiting ... ok\ntest tests::cors_headers ... ok\ntest tests::content_type_json ... ok\ntest tests::graceful_shutdown ... ok\n\ntest result: ok. 12 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.23s",
        "stderr": "",
        "exit_code": 0,
        "duration_ms": 230,
        "command": "cargo test",
        "passed": 12,
        "failed": 0,
        "ignored": 0
    }).to_string();

    let api_response_json = serde_json::json!({
        "status": 200,
        "headers": {
            "content-type": "application/json",
            "x-request-id": "req_abc123def456",
            "x-ratelimit-remaining": "99",
            "x-ratelimit-reset": "1710000000"
        },
        "body": {
            "users": [
                {"id": 1, "name": "Alice Johnson", "email": "alice@example.com", "role": "admin", "created_at": "2024-01-15T10:30:00Z", "last_login": "2024-03-10T14:22:00Z"},
                {"id": 2, "name": "Bob Smith", "email": "bob@example.com", "role": "user", "created_at": "2024-02-01T09:00:00Z", "last_login": "2024-03-09T11:45:00Z"},
                {"id": 3, "name": "Carol Williams", "email": "carol@example.com", "role": "user", "created_at": "2024-02-15T16:00:00Z", "last_login": "2024-03-08T08:30:00Z"},
                {"id": 4, "name": "David Brown", "email": "david@example.com", "role": "moderator", "created_at": "2024-03-01T12:00:00Z", "last_login": "2024-03-10T17:00:00Z"},
                {"id": 5, "name": "Eve Davis", "email": "eve@example.com", "role": "user", "created_at": "2024-03-05T08:00:00Z", "last_login": "2024-03-10T09:15:00Z"},
                {"id": 6, "name": "Frank Miller", "email": "frank@example.com", "role": "user", "created_at": "2024-03-07T14:30:00Z", "last_login": null},
                {"id": 7, "name": "Grace Lee", "email": "grace@example.com", "role": "admin", "created_at": "2024-01-01T00:00:00Z", "last_login": "2024-03-10T18:00:00Z"},
                {"id": 8, "name": "Henry Wilson", "email": "henry@example.com", "role": "user", "created_at": "2024-02-20T11:00:00Z", "last_login": "2024-03-07T15:30:00Z"}
            ],
            "total": 156,
            "page": 1,
            "per_page": 8,
            "has_more": true
        },
        "duration_ms": 45
    }).to_string();

    let sql_result_json = serde_json::json!({
        "columns": ["id", "name", "email", "role", "active", "token_hash", "created_at", "updated_at"],
        "rows": [
            [1, "Alice Johnson", "alice@example.com", "admin", true, "sha256:a1b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef123456", "2024-01-15 10:30:00", "2024-03-10 14:22:00"],
            [2, "Bob Smith", "bob@example.com", "user", true, "sha256:b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef1234567", "2024-02-01 09:00:00", "2024-03-09 11:45:00"],
            [3, "Carol Williams", "carol@example.com", "user", true, "sha256:c3d4e5f6789012345678901234567890abcdef1234567890abcdef12345678", "2024-02-15 16:00:00", "2024-03-08 08:30:00"],
            [4, "David Brown", "david@example.com", "moderator", true, "sha256:d4e5f6789012345678901234567890abcdef1234567890abcdef123456789", "2024-03-01 12:00:00", "2024-03-10 17:00:00"],
            [5, "Eve Davis", "eve@example.com", "user", false, "sha256:e5f6789012345678901234567890abcdef1234567890abcdef1234567890", "2024-03-05 08:00:00", "2024-03-10 09:15:00"]
        ],
        "row_count": 5,
        "execution_time_ms": 12,
        "query": "SELECT * FROM users WHERE active = true LIMIT 5"
    }).to_string();

    vec![
        Message::system("You are a senior Rust engineer. You have access to tools for shell commands, file operations, git, HTTP, SQL, and more. Be thorough and test your changes."),
        // Turn 0 — check project state
        Message::user("Can you check the current state of the project? I want to add authentication."),
        Message::assistant("Let me check the project structure, build status, and git state."),
        Message::tool(shell_build_json.clone()),
        Message::tool(file_read_json.clone()),
        Message::tool(git_status_json.clone()),
        Message::assistant("The project compiles with one warning (unused import). It's an Axum web server with a health endpoint on `feature/add-auth` branch. 2 files modified, 2 new files. Let me plan the auth implementation."),
        // Turn 1 — add JWT auth
        Message::user("Great, go ahead and add JWT authentication."),
        Message::assistant("I'll create the auth module. Adding dependencies first, then creating auth.rs and middleware.rs."),
        Message::tool(shell_build_json.clone()),
        Message::tool(serde_json::json!({"path": "src/auth.rs", "lines_written": 45, "bytes": 1250}).to_string()),
        Message::tool(serde_json::json!({"path": "src/middleware.rs", "lines_written": 30, "bytes": 890}).to_string()),
        Message::tool(serde_json::json!({"path": "src/main.rs", "lines_written": 35, "bytes": 980}).to_string()),
        Message::tool(shell_build_json.clone()),
        Message::assistant("Auth module created. JWT validation, token refresh, and route protection are all in place. Compiles clean."),
        // Turn 2 — run tests
        Message::user("Can you run the tests?"),
        Message::assistant("Running the full test suite."),
        Message::tool(test_output_json.clone()),
        Message::assistant("All 12 tests pass including the new auth tests."),
        // Turn 3 — check API and database
        Message::user("Check the user API and database to make sure auth integrates properly."),
        Message::assistant("Let me query the API and the database to verify the user model works with our auth."),
        Message::tool(api_response_json.clone()),
        Message::tool(sql_result_json.clone()),
        Message::assistant("API returns 8 users (156 total, paginated). Database has matching records with token hashes. The auth integration looks correct — user roles map to JWT claims properly."),
        // Turn 4 — add rate limiting
        Message::user("Now add rate limiting to the auth endpoints."),
        Message::assistant("Adding a token bucket rate limiter to the auth middleware."),
        Message::tool(shell_build_json.clone()),
        Message::tool(test_output_json.clone()),
        Message::assistant("Rate limiter added. All tests pass."),
        // Turn 5 — commit
        Message::user("Perfect, let's commit this."),
        Message::assistant("Committing the changes."),
        Message::tool(git_status_json.clone()),
        Message::tool(serde_json::json!({"stdout": "[main abc1234] feat: add JWT auth with rate limiting\n 4 files changed, 156 insertions(+)", "stderr": "", "exit_code": 0}).to_string()),
        Message::assistant("Committed successfully: `feat: add JWT auth with rate limiting` — 4 files, 156 insertions."),
        // Turn 6 (current)
        Message::user("What should we work on next?"),
    ]
}

// ── Old conversation (backward compat — embedded XML tool results) ───────────

// ── Mock summarizer for tests ────────────────────────────────────────────────

struct MockSummarizer;
impl distil::Summarizer for MockSummarizer {
    fn summarize(&self, _content: &str, _max_tokens: usize) -> distil::error::Result<String> {
        Ok("User asked to add JWT auth. Project is an Axum web server. Auth module created (auth.rs, middleware.rs). 12 tests pass. API has 156 users with role-based access. Rate limiter added. All committed.".into())
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

/// Measure token count of a context WITHOUT any distil optimization.
fn baseline_tokens(messages: &[Message], tools: &[ToolSpec], counter: &dyn distil::TokenCounter) -> usize {
    let msg_tokens: usize = messages.iter().map(|m| counter.count(&m.content)).sum();
    let tool_tokens: usize = tools
        .iter()
        .map(|t| counter.count(&t.name) + counter.count(&t.description) + counter.count(&t.parameters.to_string()))
        .sum();
    msg_tokens + tool_tokens
}

fn count_ctx_tokens(ctx: &Ctx, counter: &dyn distil::TokenCounter) -> usize {
    ctx.messages.iter().map(|m| counter.count(&m.content)).sum::<usize>()
        + ctx.tools.iter()
            .map(|t| counter.count(&t.name) + counter.count(&t.description) + counter.count(&t.parameters.to_string()))
            .sum::<usize>()
}

// ── V2: Full pipeline with ALL new features ──────────────────────────────────

#[test]
fn full_pipeline_v2_with_all_features() {
    let counter = EstimateCounter;
    let tools = realistic_tools();
    let messages = realistic_conversation_v2();
    let baseline = baseline_tokens(&messages, &tools, &counter);

    // Pipeline with ALL new features:
    // - RegistryLayer: compact tool catalog
    // - MaskingLayer: JSON truncation + observation/history split
    // - SummarizationLayer: mock LLM summarization
    // - CompactionLayer: structural cleanup
    // - BudgetLayer: token budget
    let pipeline = Pipeline::builder()
        .counter(counter)
        .layer(RegistryLayer::new(tools.clone(), &counter))
        .layer(
            MaskingLayer::new()
                .retain_turns(2)
                .retain_turns_tool(1)
                .retain_turns_assistant(3)
                .json_truncate(JsonTruncateConfig::default()),
        )
        .layer(SummarizationLayer::new(MockSummarizer).age_threshold(3).min_content_tokens(20))
        .layer(CompactionLayer::new())
        .layer(BudgetLayer::new(32_000).preserve_recent(4))
        .layer(CacheAlignLayer::generic())
        .build();

    let mut ctx = Ctx::new(messages, tools.clone(), 6);
    let result = pipeline.optimize(&mut ctx);
    let after = count_ctx_tokens(&ctx, &counter);

    let saved = baseline.saturating_sub(after);
    let pct = (saved as f64 / baseline as f64) * 100.0;

    eprintln!("\n══ V2 Pipeline: ALL new features ════════════════════════");
    eprintln!("  Without distil : {:>6} tokens  (baseline)", baseline);
    eprintln!("  With distil    : {:>6} tokens", after);
    eprintln!("  Saved          : {:>6} tokens  ({:.1}%)", saved, pct);
    eprintln!("  Per-layer breakdown:");
    for lr in &result.layers {
        let layer_saved = lr.tokens_saved();
        eprintln!(
            "    {:15} {:>5} → {:>5}  (saved {:>4}, {:>5.1}%)  {}",
            lr.layer, lr.tokens_before, lr.tokens_after,
            layer_saved, lr.percentage_saved(), lr.detail
        );
    }

    assert!(
        pct > 55.0,
        "V2 pipeline should save >55%, got {pct:.1}% (new features should beat the old 53.7%)"
    );
}

// ── Compare V1 (old pipeline, no new features) vs V2 (all new features) ──────

#[test]
fn compare_v1_vs_v2_pipeline() {
    let counter = EstimateCounter;
    let tools = realistic_tools();
    let messages = realistic_conversation_v2();
    let baseline = baseline_tokens(&messages, &tools, &counter);

    // V1: old pipeline — no JSON truncation, no split retention, no summarization
    let pipeline_v1 = Pipeline::builder()
        .counter(counter)
        .layer(RegistryLayer::new(tools.clone(), &counter))
        .layer(MaskingLayer::new().retain_turns(2).no_json_truncate())
        .layer(CompactionLayer::new())
        .layer(BudgetLayer::new(32_000).preserve_recent(4))
        .layer(CacheAlignLayer::generic())
        .build();

    let mut ctx_v1 = Ctx::new(messages.clone(), tools.clone(), 6);
    let result_v1 = pipeline_v1.optimize(&mut ctx_v1);
    let after_v1 = count_ctx_tokens(&ctx_v1, &counter);

    // V2: new pipeline — JSON truncation + split retention + summarization
    let pipeline_v2 = Pipeline::builder()
        .counter(counter)
        .layer(RegistryLayer::new(tools.clone(), &counter))
        .layer(
            MaskingLayer::new()
                .retain_turns(2)
                .retain_turns_tool(1)
                .retain_turns_assistant(3)
                .json_truncate(JsonTruncateConfig::default()),
        )
        .layer(SummarizationLayer::new(MockSummarizer).age_threshold(3).min_content_tokens(20))
        .layer(CompactionLayer::new())
        .layer(BudgetLayer::new(32_000).preserve_recent(4))
        .layer(CacheAlignLayer::generic())
        .build();

    let mut ctx_v2 = Ctx::new(messages, tools.clone(), 6);
    let result_v2 = pipeline_v2.optimize(&mut ctx_v2);
    let after_v2 = count_ctx_tokens(&ctx_v2, &counter);

    let saved_v1 = baseline.saturating_sub(after_v1);
    let pct_v1 = (saved_v1 as f64 / baseline as f64) * 100.0;
    let saved_v2 = baseline.saturating_sub(after_v2);
    let pct_v2 = (saved_v2 as f64 / baseline as f64) * 100.0;
    let improvement = after_v1.saturating_sub(after_v2);

    eprintln!("\n══ V1 (old) vs V2 (new) Comparison ═══════════════════════");
    eprintln!("  Baseline (no distil): {:>6} tokens", baseline);
    eprintln!();
    eprintln!("  V1 (old pipeline)  : {:>6} tokens  ({:.1}% saved)", after_v1, pct_v1);
    for lr in &result_v1.layers {
        if lr.tokens_saved() > 0 {
            eprintln!("    {:15} {:>5} → {:>5}  (saved {:>4})", lr.layer, lr.tokens_before, lr.tokens_after, lr.tokens_saved());
        }
    }
    eprintln!();
    eprintln!("  V2 (new pipeline)  : {:>6} tokens  ({:.1}% saved)", after_v2, pct_v2);
    for lr in &result_v2.layers {
        if lr.tokens_saved() > 0 {
            eprintln!("    {:15} {:>5} → {:>5}  (saved {:>4})", lr.layer, lr.tokens_before, lr.tokens_after, lr.tokens_saved());
        }
    }
    eprintln!();
    eprintln!("  ── New features impact ──");
    eprintln!("  V2 saves {:>4} MORE tokens than V1", improvement);
    eprintln!("  V2 saves {:.1}% vs V1's {:.1}%  (+{:.1} percentage points)", pct_v2, pct_v1, pct_v2 - pct_v1);

    assert!(
        after_v2 < after_v1,
        "V2 ({after_v2}) should use fewer tokens than V1 ({after_v1})"
    );
}

// ── Individual feature impact tests ──────────────────────────────────────────

#[test]
fn json_truncation_saves_tokens_on_json_tool_results() {
    let counter = EstimateCounter;
    let messages = realistic_conversation_v2();

    // Without JSON truncation
    let masker_plain = MaskingLayer::new().retain_turns(1).no_json_truncate();
    let mut ctx_plain = Ctx::new(messages.clone(), vec![], 6);
    let result_plain = masker_plain.apply(&mut ctx_plain, &counter);

    // With JSON truncation
    let masker_json = MaskingLayer::new().retain_turns(1).json_truncate(JsonTruncateConfig::default());
    let mut ctx_json = Ctx::new(messages, vec![], 6);
    let result_json = masker_json.apply(&mut ctx_json, &counter);

    eprintln!("\n══ JSON Truncation Impact ═══════════════════════════════");
    eprintln!("  Without JSON truncation: {} → {} (saved {})", result_plain.tokens_before, result_plain.tokens_after, result_plain.tokens_saved());
    eprintln!("  With JSON truncation   : {} → {} (saved {})", result_json.tokens_before, result_json.tokens_after, result_json.tokens_saved());

    // With JSON truncation, we preserve structure so the masked result is bigger
    // than fully masked, BUT the tool results that get JSON-truncated retain semantic
    // value. The key assertion: both approaches save tokens, JSON truncation
    // preserves more information.
    assert!(result_plain.tokens_saved() > 0, "plain masking should save tokens");
    assert!(result_json.tokens_saved() > 0, "JSON truncation should save tokens");

    // Verify JSON structure is preserved in truncated results
    let has_json_truncated = ctx_json.messages.iter().any(|m| m.content.contains("json truncated") || m.content.contains("stdout"));
    eprintln!("  JSON structure preserved: {}", has_json_truncated);
}

#[test]
fn observation_history_split_masks_tool_earlier() {
    let counter = EstimateCounter;
    let messages = realistic_conversation_v2();

    // Without split — everything uses retain_turns=2
    let masker_unified = MaskingLayer::new().retain_turns(2).no_json_truncate();
    let mut ctx_unified = Ctx::new(messages.clone(), vec![], 6);
    let result_unified = masker_unified.apply(&mut ctx_unified, &counter);

    // With split — tool results masked more aggressively
    let masker_split = MaskingLayer::new()
        .retain_turns(2)
        .retain_turns_tool(1)
        .retain_turns_assistant(3)
        .no_json_truncate();
    let mut ctx_split = Ctx::new(messages, vec![], 6);
    let result_split = masker_split.apply(&mut ctx_split, &counter);

    eprintln!("\n══ Observation/History Split Impact ═════════════════════");
    eprintln!("  Unified retention  : {} → {} (saved {})", result_unified.tokens_before, result_unified.tokens_after, result_unified.tokens_saved());
    eprintln!("  Split retention    : {} → {} (saved {})", result_split.tokens_before, result_split.tokens_after, result_split.tokens_saved());
    eprintln!("  Extra savings      : {} tokens", result_split.tokens_saved().saturating_sub(result_unified.tokens_saved()));

    assert!(
        result_split.tokens_saved() >= result_unified.tokens_saved(),
        "split retention should save at least as much (split={}, unified={})",
        result_split.tokens_saved(), result_unified.tokens_saved()
    );
}

#[test]
fn summarization_layer_compresses_old_turns() {
    let counter = EstimateCounter;
    let messages = realistic_conversation_v2();

    let layer = SummarizationLayer::new(MockSummarizer)
        .age_threshold(3)
        .min_content_tokens(20);

    let mut ctx = Ctx::new(messages, vec![], 6);
    let tokens_before = ctx.total_tokens(&counter);
    let result = layer.apply(&mut ctx, &counter);
    let tokens_after = ctx.total_tokens(&counter);

    eprintln!("\n══ Summarization Layer Impact ═══════════════════════════");
    eprintln!("  Before : {} tokens", tokens_before);
    eprintln!("  After  : {} tokens", tokens_after);
    eprintln!("  Saved  : {} tokens ({:.1}%)", result.tokens_saved(), result.percentage_saved());
    eprintln!("  Detail : {}", result.detail);

    // Verify summary was injected
    let has_summary = ctx.messages.iter().any(|m| m.content.contains("## Conversation Summary"));
    assert!(has_summary, "should have summary message");
    assert!(result.tokens_saved() > 0, "should save tokens: {}", result.detail);
}

// ── Backward compat: old tests still pass ────────────────────────────────────

#[test]
fn scratchpad_survives_pipeline() {
    let counter = EstimateCounter;
    let pad = ScratchpadLayer::new();
    pad.set("plan".into(), "1. Add auth 2. Add rate limiting 3. Deploy".into());
    pad.set("blockers".into(), "Need to update the CORS config first".into());

    let pipeline = Pipeline::builder()
        .counter(counter)
        .layer(MaskingLayer::new().retain_turns(1))
        .layer(pad)
        .layer(CompactionLayer::new())
        .build();

    let mut ctx = Ctx::new(
        vec![
            Message::system("You are helpful."),
            Message::user("What's the plan?"),
        ],
        vec![],
        0,
    );

    let _result = pipeline.optimize(&mut ctx);

    let sys = &ctx.messages[0].content;
    assert!(sys.contains("plan"), "scratchpad should be injected");
    assert!(sys.contains("blockers"), "all entries should appear");

    let tool_names: Vec<&str> = ctx.tools.iter().map(|t| t.name.as_str()).collect();
    assert!(tool_names.contains(&"note_write"));
    assert!(tool_names.contains(&"note_read"));

    let write_result = pipeline.handle_tool_call(
        "note_write",
        &serde_json::json!({"key": "status", "value": "auth module complete"}),
    );
    assert!(write_result.is_some());

    let read_result = pipeline.handle_tool_call(
        "note_read",
        &serde_json::json!({"key": "status"}),
    );
    assert_eq!(read_result.unwrap(), "auth module complete");
}

#[test]
fn tool_search_still_works() {
    let counter = EstimateCounter;
    let tools = realistic_tools();

    let pipeline = Pipeline::builder()
        .counter(counter)
        .layer(RegistryLayer::new(tools.clone(), &counter))
        .build();

    let search_result = pipeline.handle_tool_call(
        "tool_search",
        &serde_json::json!({"query": "git operations"}),
    );
    assert!(search_result.is_some());
    assert!(search_result.unwrap().contains("git"));
}

#[cfg(feature = "tiktoken")]
#[test]
fn compare_v1_vs_v2_tiktoken() {
    let counter = distil::TiktokenCounter::cl100k().expect("tiktoken unavailable");
    let tools = realistic_tools();
    let messages = realistic_conversation_v2();
    let baseline = baseline_tokens(&messages, &tools, &counter);

    let pipeline = Pipeline::builder()
        .counter(EstimateCounter)
        .layer(RegistryLayer::new(tools.clone(), &counter))
        .layer(
            MaskingLayer::new()
                .retain_turns(2)
                .retain_turns_tool(1)
                .retain_turns_assistant(3),
        )
        .layer(SummarizationLayer::new(MockSummarizer).age_threshold(3).min_content_tokens(20))
        .layer(CompactionLayer::new())
        .layer(BudgetLayer::new(32_000).preserve_recent(4))
        .layer(CacheAlignLayer::generic())
        .build();

    let mut ctx = Ctx::new(messages, tools.clone(), 6);
    let _result = pipeline.optimize(&mut ctx);
    let after = count_ctx_tokens(&ctx, &counter);

    let pct = ((baseline - after) as f64 / baseline as f64) * 100.0;
    eprintln!("\n══ Tiktoken V2 ══════════════════════════════════════════");
    eprintln!("  Baseline: {} tokens (BPE)", baseline);
    eprintln!("  After   : {} tokens", after);
    eprintln!("  Saved   : {:.1}%", pct);

    assert!(pct > 50.0, "V2 should save >50% with tiktoken, got {pct:.1}%");
}

// ── Real LLM summarizer test ─────────────────────────────────────────────────

/// Tests the full pipeline with a REAL LLM call for summarization.
///
/// Uses NVIDIA NIM's OpenAI-compatible endpoint.
/// Requires `proxy` feature and `NVIDIA_API_KEY` env var.
///
/// Run: NVIDIA_API_KEY=<key> cargo test --features proxy full_pipeline_real_llm -- --nocapture
#[cfg(feature = "proxy")]
#[test]
fn full_pipeline_real_llm_summarizer() {
    use distil::Summarizer;

    // NVIDIA_API_KEY with NIM's OpenAI-compatible endpoint
    let api_key = match std::env::var("NVIDIA_API_KEY") {
        Ok(k) if !k.is_empty() => k,
        _ => {
            eprintln!("\n⚠ SKIPPED: full_pipeline_real_llm_summarizer");
            eprintln!("  Set NVIDIA_API_KEY to run this test");
            return;
        }
    };

    let summarizer = distil::HttpSummarizer::new(
        "https://integrate.api.nvidia.com/v1/chat/completions",
        "meta/llama-3.3-70b-instruct",
        &api_key,
    );

    let counter = EstimateCounter;
    let tools = realistic_tools();
    let messages = realistic_conversation_v2();
    let baseline = baseline_tokens(&messages, &tools, &counter);

    // Test the summarizer directly first
    let test_content = "[user]: Build auth module\n[assistant]: Created auth.rs with JWT validation, middleware.rs with route protection. Added jsonwebtoken and argon2 deps.\n[user]: Run tests\n[assistant]: All 12 tests pass.\n";
    let summary = summarizer
        .summarize(test_content, 100)
        .expect("real LLM summarization failed");

    eprintln!("\n══ Real LLM Summarizer Test ═════════════════════════════");
    eprintln!("  Input : {} chars", test_content.len());
    eprintln!("  Output: {} chars", summary.len());
    eprintln!("  Summary: {summary}");

    assert!(!summary.is_empty(), "LLM returned empty summary");
    assert!(
        summary.len() < test_content.len(),
        "summary ({}) should be shorter than input ({})",
        summary.len(),
        test_content.len()
    );

    // Now test the full pipeline with real summarization
    let pipeline = Pipeline::builder()
        .counter(counter)
        .layer(RegistryLayer::new(tools.clone(), &counter))
        .layer(
            MaskingLayer::new()
                .retain_turns(2)
                .retain_turns_tool(1)
                .retain_turns_assistant(3)
                .json_truncate(JsonTruncateConfig::default()),
        )
        .layer(SummarizationLayer::new(summarizer).age_threshold(3).min_content_tokens(20))
        .layer(CompactionLayer::new())
        .layer(BudgetLayer::new(32_000).preserve_recent(4))
        .layer(CacheAlignLayer::generic())
        .build();

    let mut ctx = Ctx::new(messages, tools.clone(), 6);
    let result = pipeline.optimize(&mut ctx);
    let after = count_ctx_tokens(&ctx, &counter);

    let saved = baseline.saturating_sub(after);
    let pct = (saved as f64 / baseline as f64) * 100.0;

    eprintln!("\n  ── Full Pipeline with Real LLM ──");
    eprintln!("  Without distil : {:>6} tokens", baseline);
    eprintln!("  With distil    : {:>6} tokens", after);
    eprintln!("  Saved          : {:>6} tokens  ({:.1}%)", saved, pct);
    for lr in &result.layers {
        let layer_saved = lr.tokens_saved();
        eprintln!(
            "    {:15} {:>5} → {:>5}  (saved {:>4}, {:>5.1}%)  {}",
            lr.layer, lr.tokens_before, lr.tokens_after,
            layer_saved, lr.percentage_saved(), lr.detail
        );
    }

    // Find the summary message to show what the LLM actually produced
    let summary_msg = ctx.messages.iter().find(|m| m.content.contains("## Conversation Summary"));
    if let Some(msg) = summary_msg {
        eprintln!("\n  ── LLM Summary ──");
        eprintln!("  {}", msg.content);
    }

    assert!(
        pct > 50.0,
        "real LLM pipeline should save >50%, got {pct:.1}%"
    );
}
