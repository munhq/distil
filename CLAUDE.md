# Distil — Context optimization for LLM agents

## What This Is
Rust crate + HTTP server that reduces token usage in LLM agent conversations by 50-90%.
Composable pipeline of optimization layers, each independently testable.

Two consumption modes:
- **Crate**: `distil = { path = "../distil" }` for Rust projects
- **HTTP server**: `POST /v1/optimize` for any language/framework

## Architecture

```
Agent Loop → Pipeline[Registry → Masking → Summarization → Compaction → Budget → Cache] → LLM
```

Each layer implements the `Layer` trait from `src/pipeline.rs`:
- `apply(&self, ctx: &mut Ctx, counter: &dyn TokenCounter) -> LayerResult`
- `handle_tool_call(name, args) -> Option<String>` (for layers that inject tools)

## Layers (src/layers/)

| Layer | File | What |
|-------|------|------|
| RegistryLayer | registry_layer.rs | Compact tool catalog + `tool_search` meta-tool |
| MaskingLayer | masking_layer.rs | Replace old tool outputs with summaries, JSON truncation, split retention |
| SummarizationLayer | summarization_layer.rs | LLM-based semantic compression via caller-provided `Summarizer` trait |
| CompactionLayer | compactor_layer.rs | Dedup, whitespace strip, merge consecutive messages |
| BudgetLayer | budget_layer.rs | Trim oldest messages to fit token budget |
| ScratchpadLayer | scratchpad_layer.rs | Agent working memory outside context window |
| CacheAlignLayer | cache_layer.rs | Reorder system prompt for prompt cache hits |

## Business Logic Modules (src/)

| Module | What |
|--------|------|
| registry.rs | ToolRegistry: catalog generation, keyword search, relevance scoring |
| masker.rs | ResultMasker: regex-based detection, JSON truncation, observation/history split |
| summarizer.rs | `Summarizer` trait for caller-provided LLM summarization |
| budget.rs | TokenBudget: token breakdown analysis, message trimming |
| counter.rs | TokenCounter trait + EstimateCounter (chars/3.5 heuristic) |
| pipeline.rs | Layer trait, Pipeline, Ctx, PipelineResult |
| types.rs | Message, Role, ToolSpec, ToolSummary, Stats |
| error.rs | Error types |

## Key Design Decisions

- **LLM-agnostic core**: distil's structural layers never call an LLM.
- **SummarizationLayer**: the one exception — caller provides the LLM via `Summarizer` trait.
- **Sync by design**: all layers are sync. Async callers use `block_in_place`.
- **Framework-agnostic**: Messages are just `{role, content}` strings. Works with any agent.
- **Composable**: each layer works independently or composed in any order.
- **Measurable**: every layer reports tokens_before, tokens_after, and detail.
- **Dual consumption**: Rust crate for native integration, HTTP server for polyglot teams.

## MaskingLayer Features

- **JSON truncation** (default on): when tool results are JSON, keeps structure (keys, types) but truncates long strings, large arrays, deep nesting. Preserves semantic signal. Config via `JsonTruncateConfig`.
- **Observation/history separation**: `retain_turns_tool` (aggressive) vs `retain_turns_assistant` (conservative). Tool results are pure data; assistant reasoning is what the LLM needs to follow its logic.
- **Pattern-based**: XML tags `<tool_result>` and bracket `[Tool:]` patterns, or custom regex.

## HTTP Server (src/bin/proxy.rs)

Build: `cargo build --features proxy`
Run: `distil-proxy --port 8080` (direct mode) or `--upstream https://api.openai.com/v1` (proxy mode)

### Endpoints

| Method | Path | What |
|--------|------|------|
| POST | /v1/optimize | Direct: send messages+tools, get optimized context + metrics |
| POST | /v1/tool_call | Handle distil-injected tools (tool_search, note_read, note_write) |
| GET | /v1/health | Health check with mode/version/config |
| POST | /v1/chat/completions | Proxy: optimize + forward to upstream LLM |

### Environment Variables

- `DISTIL_UPSTREAM` — upstream LLM API base URL (enables proxy mode)
- `DISTIL_PORT` — listen port (default 8080)
- `DISTIL_BUDGET` — token budget (default 32000)
- `DISTIL_MODEL` — model name for token counting
- `DISTIL_SUMMARIZER_ENDPOINT` — LLM endpoint for SummarizationLayer
- `DISTIL_SUMMARIZER_MODEL` — model for summarization
- `DISTIL_SUMMARIZER_API_KEY` — API key for summarizer

## Testing

- Unit tests in each module (48 tests)
- Integration test in `tests/pipeline_integration.rs` with realistic 30-tool scenario
- Run: `cargo test`
- With tiktoken: `cargo test --features tiktoken`

## Benchmarks (realistic 30-tool, 5-turn conversation)

```
Distil: 4,065 → 1,820 tokens (55.2% savings)
  registry: 30% savings (1,939 → 627 tool tokens)
  masking:  35% savings (old tool results)
  compactor: 1% savings (whitespace)
```
