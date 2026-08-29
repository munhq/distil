# A/B protocol: does codeindex reduce tokens on the same task?

The corpus split cannot answer this. Only 32 sessions in 13,814 ever called a
codeindex tool, all of them long, so session length produces the entire apparent
effect. Holding the task fixed and moving only the tool set is the way out.

## Run it

Pick a task that needs code comprehension and no writes, so both windows do
comparable work. Something like:

    "Explain how session parsing works in this repo: which function parses a
     transcript, what calls it, and what breaks if its signature changes.
     Do not edit any files."

Start from a clean tree at a known commit, and run the two windows one after the
other, not at once — concurrent runs contend for CPU and distort the timings.

Window A, codeindex available. The config launches `codeindex` from `PATH`, so
install it first (`npx -y @munhq/codeindex`, or `install.sh` from that repo) or
edit the `command` field to point at your own build:

    claude --strict-mcp-config --mcp-config bench/ab/with-codeindex.json

Window B, no MCP servers at all:

    claude --strict-mcp-config --mcp-config bench/ab/without-codeindex.json

Paste the SAME prompt verbatim into both. Let each run to completion.

## Measure

    python bench/ab_windows.py <A.jsonl> <B.jsonl>

With no arguments it takes the two most recently modified transcripts, which is
usually what you want straight after the runs.

## Hold these fixed, or the number is worthless

1. The same prompt, pasted, not retyped.
2. The same repository at the same commit, tree clean.
3. The same model in both windows.
4. Both runs finish; neither is interrupted.
5. The skill is installed for A if you are testing the skill, and absent from
   both if you are testing the server alone. Those are different experiments.

## Read the result honestly

Compare **per turn**, not in total: the two runs will not take the same number
of turns, and totals are not comparable until that is divided out. The harness
prints both.

An agent is not deterministic. One pair is an anecdote. Run at least five pairs,
alternating which window goes first, before believing a direction — and expect
the variance between identical runs to be large enough to swallow a small
effect.
