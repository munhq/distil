# Contributing to distil

Thanks for your interest. This guide covers building, testing and submitting
changes.

## Read this first

distil exists because of one measurement, and it constrains what belongs here:
rewriting a conversation's history invalidates the prompt cache from the edit
onwards, converting cache reads billed at 0.1x into cache writes billed at 1.25x
or 2.0x. On real agent traffic that costs more than the compression saves.

`README.md` carries the numbers and the break-even table. A change that rewrites
history by default has to be priced against that table in the pull request, not
justified by a savings percentage alone. A savings percentage with no cache
accounting is the thing this project was built to argue against.

## Prerequisites

- Rust 1.85 or later (the crate is edition 2024)
- Node 18 or later, only if you touch `npm/`
- `zip`, only if you build the `.mcpb` bundle

## Build and test

```bash
git clone https://github.com/munhq/distil.git
cd distil

cargo build                                                  # bare crate
cargo test --features "bench probe mcp metrics config"       # 136 tests
cargo clippy --all-targets --features "bench probe mcp metrics config" -- -D warnings
cargo fmt --all --check
```

CI runs all four, then repeats the build with each feature alone, because
feature combinations break one at a time and the combined build hides that. It
also builds the shipped binaries on Linux, macOS and Windows.

Run the binaries:

```bash
cargo run --features mcp   --bin distil-mcp                  # MCP server, stdio
cargo run --features proxy --bin distil-proxy -- --port 8080 # HTTP server
cargo run --features bench --release --bin distil-bench -- ~/.claude/projects
```

## Project structure

```
src/            the crate: pipeline, layers, counters, registry, masker, probe
src/layers/     one file per Layer implementation
src/bin/        distil-mcp, distil-bench, distil-probe, distil-proxy
tests/          integration tests against a realistic 30-tool conversation
bench/          measurement harness, protocol and recorded results
npm/            the @munhq/distil wrapper, the .mcpb builder, the Smithery publisher
plugin/         the Claude Code plugin: skill, launcher, install tests
```

## Adding a layer

1. Add `src/layers/<name>_layer.rs` and implement `Layer` from `src/pipeline.rs`.
2. Report `tokens_before`, `tokens_after` and a detail line. Every layer must
   be measurable on its own.
3. Keep it sync. Every layer is sync; async callers use `block_in_place`.
4. Do not call an LLM from the layer. If you need one, take it through the
   `Summarizer` or `Completer` trait so the caller supplies it.
5. Add unit tests in the same file, and export the layer from `src/layers/mod.rs`.

## Benchmark rules

If your change touches `bench/`, or you quote a number from it:

1. One tokenizer for every measurement. A tool never scores itself.
2. Feed whole sessions in the recorded wire format, never isolated payloads.
   A payload-level harness measures the harness.
3. Report what a tool does by default AND with its limits unlocked.
4. Report the index of the first message a compressor edits. Without it a
   savings figure says nothing about cost.

`bench/README.md` explains each rule and names the two harness mistakes that
produced wrong numbers before they were caught.

## Committed files must not carry personal data

Never commit an absolute home path, a personal account name, or the name of a
private project — in code, tests, fixtures, comments, docs or recorded results.
Use `~`, `owner/repo`, `example-app` or a placeholder. `.codeindex.json` is
generated and gitignored for exactly this reason.

## Commit messages

One line, a lowercase type prefix, then what changed and why it needed changing:

```
fix: a relative --out was created inside a temp dir that is deleted
feat: publish the Smithery listing from code, not from a terminal
```

Describe the defect, not the patch. Keep the subject under 72 characters.

## Pull requests

1. Open an issue first for anything that changes behaviour or adds a dependency.
2. Keep the diff to one subject.
3. Make CI green before asking for review.
4. Update `README.md` or `CLAUDE.md` when you change behaviour they describe.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

Unless you explicitly state otherwise, any contribution you intentionally submit
for inclusion in this work, as defined in the Apache-2.0 license, shall be dual
licensed as above, without any additional terms or conditions.
