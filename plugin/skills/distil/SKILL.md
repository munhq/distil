---
name: distil
description: >-
  Decide whether to compress an agent's context, and compress it when the answer
  is yes. Use when a conversation is approaching the context window limit, when a
  request has failed or is about to fail for length, or when the user asks to
  compact, shrink or trim the context. Also use to measure where a session's
  tokens actually go. Backed by the distil MCP server.
---

# Compressing context with distil

## Read this before compressing anything

Compressing a conversation usually costs more than it saves, and the reason is
prompt caching, not compression quality.

A cached prefix is billed at 0.1x. Editing history invalidates that prefix from
the edit onwards, so every token after the edit is re-billed as a cache write at
1.25x or 2.0x. Measured across 13,694 real sessions: cache writes were 2.1% of
input tokens and **28.0% of input cost**.

So a rewrite has to shrink what it invalidates below this to break even on price
alone, before any quality loss:

| turns remaining | 5m TTL | 1h TTL |
|---|---|---|
| 1 | 8.0% | 5.0% |
| 10 | 46.5% | 34.5% |
| 20 | 63.5% | 51.3% |
| 100 | 89.7% | 84.0% |

Per unit of history at 10 remaining turns: keeping it costs 1.00, compressing it
costs 2.15, and never admitting it costs 0.

**So do not compress a healthy conversation.** It is not a free optimisation and
it is not a tidy-up.

## When to use `optimize`

One situation makes it unambiguously correct: **the context window is about to
overflow.** There the comparison is no longer cheap against expensive, it is
expensive against a failed request, and cache price stops mattering.

Reach for `mcp__distil__optimize` when:

1. A request has failed for context length, or the window is nearly full.
2. The user explicitly asks to compact, shrink or trim the conversation.
3. A long-running session must continue and there is no room left.

Pass the messages, any tool definitions, and a `budget`. The result carries the
optimized context and a per-layer breakdown of what each layer saved.

If distil injects meta-tools (`tool_search`, `note_read`, `note_write`), route
those calls back through `mcp__distil__tool_call`.

## When NOT to use it

1. **Routine token saving mid-conversation.** The table above says this loses.
2. **Near the end of a task.** With few turns left a rewrite must delete over
   90% of history merely to break even.
3. **To clear old tool results on the Claude API.** Prefer the provider's
   `clear_tool_uses` context editing: it runs server-side, it takes
   `clear_at_least` so a clear only fires when it is large enough to pay for the
   invalidation, and it is one API parameter rather than a dependency.
4. **To shrink tool schemas.** Prefer Anthropic's Tool Search Tool, which is
   native and does the same job.

## The cheaper move, almost always

A token never admitted is billed zero times. A token admitted and later
compressed has already been re-sent on every turn in between, and the edit that
removes it invalidates the cache.

So before compressing, ask whether the context needed to be that large. In this
corpus `Read` was 61.4% of all tool-result tokens at 1,564 tokens per call,
while a structural lookup answering the same question cost 35 to 113. Not
reading the file beats compressing it afterwards, by a wide margin and with no
cache penalty.

## Measuring

The MCP server optimizes. The measurement lives in the CLI, which a person runs:

```bash
distil-bench ~/.claude/projects            # where tokens go, and what they cost
distil-bench ~/.claude/projects --split-by-tool mcp__codeindex__
```

Report the per-layer breakdown whenever you optimize, so the user can see what
was spent and what was actually saved.
