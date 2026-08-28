# distil — context optimization for LLM agents

## What this is

A Rust crate and a set of binaries that measure where an agent conversation's
tokens go, and compress that conversation only where compression pays for the
prompt cache it invalidates.

Read `README.md` before changing anything here. It carries the measurement that
shapes every design decision: a history rewrite invalidates the cached prefix
from the edit onwards, converting cache reads at 0.1x into cache writes at 1.25x
or 2.0x, and on real traffic that costs more than the rewrite saves. Do not add
a layer or a default that rewrites history without pricing it against that table.

## Consumption modes

1. **Crate** — `distil = "0.3"`, or a path dependency. Pure library, no binary.
2. **MCP server** — `distil-mcp`, shipped as `@munhq/distil` on npm, as a Claude
   Code plugin, and as release binaries for six targets.
3. **HTTP server** — `distil-proxy`, direct (`POST /v1/optimize`) or as a proxy
   in front of an upstream LLM API.

## Architecture

```
Ctx (messages + tools + turn)
  -> Pipeline [ Layer, Layer, ... ]
  -> PipelineResult (tokens_before, tokens_after, per-layer detail)
```

Every layer implements `Layer` from `src/pipeline.rs`:

- `apply(&self, ctx: &mut Ctx, counter: &dyn TokenCounter) -> LayerResult`
- `handle_tool_call(name, args) -> Option<String>` for layers that inject tools

## Layers (`src/layers/`)

| Layer | File | What it does | Touches history |
|---|---|---|---|
| CacheAlignLayer | cache_layer.rs | Orders content so the stable prefix stays cacheable | no |
| ScratchpadLayer | scratchpad_layer.rs | Keeps working state outside the context window | no |
| RegistryLayer | registry_layer.rs | Compact tool catalog plus a `tool_search` meta-tool | no |
| CodeModeLayer | code_mode_layer.rs | JS sandbox that chains several tools in one call | no |
| BudgetLayer | budget_layer.rs | Trims oldest messages to fit a token budget | yes |
| MaskingLayer | masking_layer.rs | Replaces old tool results with summaries, truncates JSON | yes |
| SummarizationLayer | summarization_layer.rs | LLM-based semantic compression | yes |
| CompactionLayer | compactor_layer.rs | Dedup, whitespace strip, merges adjacent messages | yes |

The last four are for one boundary only: **context overflow**, where the
alternative is a failed request and cache price stops being the comparison.
`RegistryLayer` and `CodeModeLayer` predate Anthropic's Tool Search Tool and
Programmatic Tool Calling; prefer the native features.

## Modules (`src/`)

| Module | What |
|---|---|
| pipeline.rs | `Layer` trait, `Pipeline`, `Ctx`, `PipelineResult` |
| types.rs | `Message`, `Role`, `ToolSpec`, `ToolSummary`, `Stats` |
| counter.rs | `TokenCounter` trait, `EstimateCounter`, `TiktokenCounter` |
| registry.rs | `ToolRegistry`: catalog generation, keyword search, scoring |
| masker.rs | `ResultMasker`: regex detection, JSON truncation, split retention |
| budget.rs | `TokenBudget`: breakdown analysis and message trimming |
| summarizer.rs | `Summarizer` and `Completer` traits, HTTP and Ollama backends |
| corpus.rs | Loader for real transcripts (`~/.claude/projects/**/*.jsonl`) |
| probe.rs | Retention probes: recall, artifact, continuation, decision |
| config.rs | TOML pipeline configuration |
| http.rs | Shared HTTP client for the summarizer and the judge |
| metrics.rs | Prometheus collectors |
| error.rs | Error types |

## Binaries (`src/bin/`), each feature-gated

| Binary | Feature | What |
|---|---|---|
| distil-mcp | `mcp` | MCP server over stdio |
| distil-bench | `bench` | Measures a corpus: segments, cache cost, break-even |
| distil-probe | `probe` | LLM-graded retention on a compressed session |
| distil-proxy | `proxy` | HTTP server, direct or in front of an upstream API |

## Design decisions

- **The core never calls an LLM.** `SummarizationLayer` is the one exception,
  and the caller supplies the LLM through the `Summarizer` trait.
- **Sync by design.** Every layer is sync. Async callers use `block_in_place`.
- **Framework-agnostic.** A `Message` is `{role, content}`. No SDK types leak in.
- **Composable.** Each layer runs alone or in any order.
- **Measurable.** Every layer reports `tokens_before`, `tokens_after` and a
  detail line, so it can be measured on its own.
- **No self-scored numbers.** A tool never scores itself, and one tokenizer is
  used for every measurement. See `bench/README.md` for the fairness rules.

## Testing

```bash
cargo test --features "bench probe mcp metrics config"   # 136 tests
cargo clippy --all-targets --features "bench probe mcp metrics config" -- -D warnings
cargo fmt --all --check
cargo build --no-default-features                        # the crate must build bare
```

CI additionally builds each feature in isolation, builds the shipped binaries on
Linux, macOS and Windows, and runs `plugin/test_platform.sh` so the asset names
the installers ask for cannot drift from the names the release publishes.

## Release

Three files carry the version and `mcp-registry.yml` refuses to publish unless
they agree: `Cargo.toml`, `npm/package.json` and `server.json` (twice — the
server version and the package version). `npm/package.json` also carries
`mcpName`, which must equal `server.json`'s `name`.

A version tag `v*` fires four workflows: `release` (six targets plus the `.mcpb`
bundle and `checksums.txt`), `release-npm`, `mcp-registry` and `smithery`.

## Conventions

- Comments say **why**, not what. A comment that restates the line is noise.
- Do not put an absolute path, a personal account name or a private project name
  into a committed file. Use `~`, `owner/repo` or a placeholder.
- `.codeindex.json` is generated and gitignored; never commit it.
