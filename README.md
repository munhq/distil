# distil

Measure what context compression actually costs, on real agent traffic.

Every tool in this space publishes a savings percentage measured on its own
fixtures. What none publishes is the denominator: what share of a real session
it is allowed to touch, and what that share costs once prompt-cache pricing is
applied. `distil` measures both on transcripts an agent actually wrote.

**What is prior art, and what is not.** The cache arithmetic below is not a
discovery. Anthropic's [context editing
docs](https://platform.claude.com/docs/en/build-with-claude/context-editing)
state that clearing tool results invalidates the cached prefix, and ship
`clear_at_least` so a clear only fires when it is large enough to pay for that.
The break-even rule is published too: on a 5-minute cache, cleared tokens times
requests-before-the-next-clear must exceed 11.5 times the tokens you keep. The
table in this README reproduces that rule exactly — it was derived
independently, which is a check on the arithmetic, not a contribution.

The gap is empirical. Every source says to calibrate against your own workload,
and none ships a way to do it or publishes what the values turn out to be. That
is what this crate is for: measuring the numbers you need in order to choose
`clear_at_least`, or to decide not to clear at all.

## The result that shapes the rest

Measured over 13,681 local Claude Code transcripts — 366,008 assistant turns,
195M tokens of unique text:

| | tokens | share |
|---|---|---|
| tool results | 117,460,908 | 60.1% |
| tool calls | 32,391,631 | 16.6% |
| user text | 29,363,474 | 15.0% |
| assistant text | 14,187,961 | 7.3% |
| thinking | 2,076,716 | 1.1% |

Those 195M unique tokens were billed as **100.6 billion input tokens** — every
token paid for 515 times, because a request resends the whole history.

Price that at real cache multipliers (read 0.1x, write 1.25x for the 5-minute
TTL and 2.0x for the 1-hour):

| | share of tokens | share of **cost** |
|---|---|---|
| cache read | 97.8% | 71.1% |
| cache writes | **2.1%** | **28.1%** |

**Cache writes are 2% of the tokens and 28% of the bill.** Editing history
invalidates the cached prefix from the edit onwards, converting reads at 0.1x
into writes at 1.25x or 2.0x. So a rewrite must shrink what it invalidates below:

| turns remaining | 5m TTL | 1h TTL |
|---|---|---|
| 1 | 8.0% | 5.0% |
| 10 | 46.5% | 34.5% |
| 20 | 63.5% | 51.3% |
| 100 | 89.7% | 84.0% |

That table is the published break-even rule in another form: at every row,
`cleared x turns / kept` equals 11.5 for the 5-minute tier. Use it to pick a
`clear_at_least` value, and use `distil-bench` to find the turn count and tail
size to put into it — those are workload properties, and they are the part
nobody publishes.

Per unit of history, at 10 remaining turns: keeping it costs 1.00, compressing
it costs 2.15, and never admitting it costs 0. **You cannot compress your way
out of context cost. You can only decline to admit tokens.**

## What that means for using this crate

Layers that do not touch history are on the right side of that arithmetic:
`CacheAlignLayer` (orders content so the stable prefix stays cacheable) and
`ScratchpadLayer` (keeps working state outside the window).

Layers that rewrite history — `MaskingLayer`, `SummarizationLayer`,
`CompactionLayer` — cost more than they save in the common case. Reach for them
at one boundary only: **context overflow**, where the alternative is a failed
request and cache price stops being the comparison. `BudgetLayer` exists for
exactly that moment.

`RegistryLayer` and `CodeModeLayer` predate Anthropic's Tool Search Tool and
Programmatic Tool Calling, which do the same jobs natively and better. Prefer
the native features.

For clearing old tool results, prefer the provider's `clear_tool_uses` context
editing over `MaskingLayer`: it runs server-side, it takes `clear_at_least`, and
it is one API parameter against a dependency. Reach for a layer here only when
you need behaviour the API does not offer.

## Measuring

```bash
cargo build --features bench --release

# Where tokens are, what they cost, and the break-even table
./target/release/distil-bench ~/.claude/projects --json baseline.json

# Sessions that called a given tool, against those that did not
./target/release/distil-bench ~/.claude/projects --split-by-tool mcp__codeindex__

# Export real traffic so other compressors run on the same input
./target/release/distil-bench ~/.claude/projects --export-sessions ./sessions --min-turns 40
```

See [`bench/README.md`](bench/README.md) for the external-tool comparison, the
fairness rules, and the two harness mistakes that produced wrong numbers first.

## Retention

A saving is only a saving if the model can still answer what the original
context could answer.

```bash
# No LLM judge: file paths checked against ground truth from the transcript
python bench/artifact_retention.py ./sessions 12

# LLM-graded probes (recall / artifact / continuation / decision)
cargo build --features probe --release
./target/release/distil-probe <session.jsonl> --probes 6 --model qwen2.5:3b
```

The probe taxonomy is [Factory.ai's](https://factory.ai/news/evaluating-compression);
their write-up defines it and ships no harness. The judge is a `Completer`,
never a `Summarizer` — a summarizer may impose summarization framing, which
rewrites both the probe format and the grading instruction.

## Using it as a library

```rust
use distil::{CacheAlignLayer, Ctx, EstimateCounter, Pipeline};

let pipeline = Pipeline::builder()
    .counter(EstimateCounter)
    .layer(CacheAlignLayer::generic())
    .build();

let mut ctx = Ctx::new(messages, tools, turn);
let result = pipeline.optimize(&mut ctx);
println!("{result}");
```

This example is kept compilable as
[`examples/readme_quickstart.rs`](examples/readme_quickstart.rs) — run it with
`cargo run --example readme_quickstart`.

Every layer implements `Layer` and reports `tokens_before`, `tokens_after` and a
detail line, so each one can be measured on its own.

## Features

| feature | what it adds |
|---|---|
| `corpus` | transcript loader (no extra dependencies) |
| `bench` | `distil-bench`, needs `tiktoken` |
| `probe` | `distil-probe`, needs `proxy` for the HTTP judge |
| `tiktoken` | accurate BPE counts instead of the chars/3.5 estimate |
| `proxy` | `distil-proxy` HTTP server |
| `mcp` | `distil-mcp` MCP server |
| `metrics` | Prometheus `/metrics` |

## Caveats

The corpus is one developer's machine. The **ratios** are the finding; the
absolute totals are personal. Counts use `cl100k_base`, which approximates
Claude's tokenizer within a few percent. The break-even model assumes a single
cache breakpoint, so a rewrite confined to the tail costs less than the table
shows — that refines it, it does not reverse it.

## License

MIT OR Apache-2.0.
