# distil-bench — compression measured on real agent traffic

Every context compressor publishes savings measured on its own fixtures. None
publish the denominator: what share of a real session they are allowed to touch,
and what that share costs once prompt-cache pricing is applied.

This harness measures both, on transcripts an agent actually wrote to disk.

## 1. Baseline — where tokens are and what they cost

```bash
cargo build --features bench --release
./target/release/distil-bench ~/.claude/projects --json baseline.json
```

Reports a local tiktoken count per segment class and, separately, the tokens the
provider billed. The two answer different questions and are never blended.

## 2. Compare against external compressors

```bash
./target/release/distil-bench ~/.claude/projects \
    --export-sessions ./sessions --min-turns 40 --max-sessions 80
python bench/session_compare.py ./sessions 40
```

Sessions are exported in the recorded wire format, verbatim. A compressor routes
on message role and block structure, so rebuilding that from flattened segments
would benchmark the rebuild.

**Feed whole sessions, never isolated payloads.** Headroom protects the last four
messages, skips content under 250 tokens, and by default does not compress user
messages — and in the Anthropic wire format tool results ARE user messages. A
payload-level harness measures the harness. An earlier version of this one did,
and its numbers were discarded.

`session_compare.py` also reports the index of the first message each compressor
modifies. Prompt caching bills a matched prefix at 0.1x, so an edit near the head
of a conversation invalidates everything after it and converts those reads into
writes at 1.25x-2.0x.

## 3. Compare structural lookup against file reads

```bash
./target/release/distil-bench ~/.claude/projects --export-payloads p.jsonl --per-tool 800
python bench/codeindex_compare.py p.jsonl ~/.local/bin/codeindex 200
```

Takes real `Read` calls whose file still exists and asks codeindex the structural
equivalent. Read and get_outline do NOT answer the same question: Read truncates
long files, while an outline enumerates every symbol, so on large files the
outline is larger. Treat the per-call figures from the corpus itself as the
trustworthy comparison and this script as a bound.

## Fairness rules

1. One tokenizer for every measurement. A tool never scores itself.
2. Population reweighting. Tool output is skewed enough that an unweighted mean
   over 171 tools describes a corpus nobody has.
3. Tool calls, not tool mentions. A grep for `mcp__codeindex__` matches 842
   sessions; only 28 ever called one. Availability is not use.
4. Report what a tool does by default AND with its limits unlocked. Either alone
   misleads in a different direction.

## Measured result — Headroom 0.35.0 on 12 real sessions

| config | tokens | saved | sessions modified | prefix surviving | tail shrank to |
|---|---|---|---|---|---|
| default | 1,113,048 → 1,023,876 | 8.0% | 12/12 | 15.3% | 90.5% |
| `compress_user_messages=True` | 1,113,048 → 1,022,594 | 8.1% | 12/12 | 14.9% | 90.5% |

Unlocking user-message compression changed the outcome by 0.1 points, so the
conservative default is not what bounds the result.

The first edit lands at a median of 10% through the conversation. That means the
saving is bought by invalidating roughly 85% of the cached prefix in order to
remove 9.5% of it. Priced at cache rates (read 0.1x, write 1.25x), one request
after such a rewrite costs about 0.97 per unit of history against 0.10 for
leaving it alone, and the rewrite only repays itself after roughly 110 further
turns — beyond the 99th percentile of session length in this corpus.

**This is a retrospective application and must be read as one.** Headroom is
built to run as a live proxy, compressing content as it ages out of the live
zone, so in production the edit point advances gradually and the cache damage is
spread across turns rather than taken at once. The arithmetic facing each
individual edit is the same either way; the aggregate is not.

The general result stands independently of any one tool: on this corpus a
history rewrite must shrink the invalidated tail below 46.5% (10 turns
remaining), 63.5% (20) or 81.3% (50) to break even. Measured shrink was 90.5%.
