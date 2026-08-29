#!/usr/bin/env python3
"""Compare two Claude Code sessions — the A/B a person runs in two windows.

Everything else in this harness measures history. This measures an experiment
you deliberately run: same task, two windows, one tool difference.

    window A   codeindex registered, skill installed
    window B   codeindex disabled

Both windows write a transcript to ~/.claude/projects. Point this at the two
files and it reports where the tokens went in each.

WHY A CONTROLLED PAIR AND NOT THE CORPUS
The observational split in distil-bench is confounded beyond rescue: only 32
sessions in 13,814 ever called codeindex, all of them long, and session length
produces the entire apparent effect. Two windows on one task removes that,
because the task is held fixed and the tool set is the only thing that moves.

WHAT TO HOLD FIXED, OR THE RESULT IS WORTHLESS
  - the same prompt, pasted verbatim into both windows
  - the same repository, at the same commit, with a clean tree
  - the same model
  - both windows run to completion, neither abandoned early
Two runs are never identical — an agent is not deterministic — so treat a
single pair as an anecdote and run several pairs before believing a number.
"""
import glob
import json
import os
import sys
from collections import Counter

try:
    import tiktoken
    ENC = tiktoken.get_encoding("cl100k_base")
    def ntok(s):
        return len(ENC.encode(s or "", disallowed_special=()))
except ImportError:                       # estimate rather than refuse to run
    def ntok(s):
        return len(s or "") // 4


def load(path):
    """Token totals, tool calls and billed usage for one transcript."""
    kinds = Counter()
    tools = Counter()
    tool_result_tokens = Counter()
    billed = Counter()
    turns = 0
    names = {}                            # tool_use_id -> tool name

    for line in open(path, errors="replace"):
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("type") not in ("user", "assistant"):
            continue
        m = d.get("message")
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role == "assistant":
            turns += 1
            u = m.get("usage") or {}
            billed["input"] += u.get("input_tokens", 0)
            billed["cache_write"] += u.get("cache_creation_input_tokens", 0)
            billed["cache_read"] += u.get("cache_read_input_tokens", 0)
            billed["output"] += u.get("output_tokens", 0)
            cc = u.get("cache_creation") or {}
            billed["w5m"] += cc.get("ephemeral_5m_input_tokens", 0)
            billed["w1h"] += cc.get("ephemeral_1h_input_tokens", 0)

        c = m.get("content")
        if isinstance(c, str):
            kinds["user_text" if role == "user" else "assistant_text"] += ntok(c)
            continue
        if not isinstance(c, list):
            continue
        for b in c:
            if not isinstance(b, dict):
                continue
            t = b.get("type")
            if t == "text":
                kinds["user_text" if role == "user" else "assistant_text"] += ntok(b.get("text", ""))
            elif t == "thinking":
                kinds["thinking"] += ntok(b.get("thinking", ""))
            elif t == "tool_use":
                name = b.get("name", "unknown")
                tools[name] += 1
                if b.get("id"):
                    names[b["id"]] = name
                kinds["tool_use"] += ntok(json.dumps(b.get("input") or {}))
            elif t == "tool_result":
                name = names.get(b.get("tool_use_id"), "unknown")
                content = b.get("content")
                text = ""
                if isinstance(content, str):
                    text = content
                elif isinstance(content, list):
                    text = "\n".join(x.get("text", "") for x in content
                                     if isinstance(x, dict) and x.get("type") == "text")
                n = ntok(text)
                kinds["tool_result"] += n
                tool_result_tokens[name] += n

    return {
        "path": path, "turns": turns, "kinds": kinds, "tools": tools,
        "results": tool_result_tokens, "billed": billed,
        "total": sum(kinds.values()),
    }


def price_units(b):
    """Input re-expressed in multiples of the base input price."""
    w5, w1h = b["w5m"], b["w1h"]
    if w5 + w1h == 0:                     # no TTL split recorded
        w5 = b["cache_write"]
    return b["input"] + b["cache_read"] * 0.1 + w5 * 1.25 + w1h * 2.0


def delta(a, b):
    return f"{(a - b) / b * 100:+.1f}%" if b else "n/a"


def main():
    if len(sys.argv) >= 3:
        pa, pb = sys.argv[1], sys.argv[2]
    else:
        files = sorted(glob.glob(os.path.expanduser(
            "~/.claude/projects/*/*.jsonl")), key=os.path.getmtime)[-2:]
        if len(files) < 2:
            print("need two transcripts")
            sys.exit(1)
        pa, pb = files[-1], files[-2]
        print("no paths given; using the two most recently modified transcripts\n")

    A, B = load(pa), load(pb)
    print(f"A = {os.path.basename(A['path'])}")
    print(f"B = {os.path.basename(B['path'])}")
    print(f"\n{'metric':<26} {'A':>14} {'B':>14} {'A vs B':>10}")
    print(f"{'assistant turns':<26} {A['turns']:>14} {B['turns']:>14} "
          f"{delta(A['turns'], B['turns']):>10}")

    for k in ("tool_result", "tool_use", "assistant_text", "user_text", "thinking"):
        print(f"{k:<26} {A['kinds'][k]:>14} {B['kinds'][k]:>14} "
              f"{delta(A['kinds'][k], B['kinds'][k]):>10}")
    print(f"{'TOTAL text tokens':<26} {A['total']:>14} {B['total']:>14} "
          f"{delta(A['total'], B['total']):>10}")

    ua, ub = price_units(A["billed"]), price_units(B["billed"])
    print(f"\n{'billed input tokens':<26} "
          f"{A['billed']['input'] + A['billed']['cache_write'] + A['billed']['cache_read']:>14} "
          f"{B['billed']['input'] + B['billed']['cache_write'] + B['billed']['cache_read']:>14}")
    print(f"{'input PRICE units':<26} {ua:>14.0f} {ub:>14.0f} {delta(ua, ub):>10}")
    print(f"{'output tokens':<26} {A['billed']['output']:>14} "
          f"{B['billed']['output']:>14} {delta(A['billed']['output'], B['billed']['output']):>10}")

    # Per turn, because the two runs will not take the same number of turns and
    # the totals are not comparable until that is divided out.
    print(f"\n{'--- per assistant turn ---':<26}")
    for label, key in (("text tokens/turn", "total"), ):
        a = A[key] / max(A["turns"], 1)
        b = B[key] / max(B["turns"], 1)
        print(f"{label:<26} {a:>14.1f} {b:>14.1f} {delta(a, b):>10}")
    a = ua / max(A["turns"], 1)
    b = ub / max(B["turns"], 1)
    print(f"{'price units/turn':<26} {a:>14.1f} {b:>14.1f} {delta(a, b):>10}")

    print(f"\n{'--- tool calls ---':<26}")
    allt = sorted(set(A["tools"]) | set(B["tools"]),
                  key=lambda t: -(A["tools"][t] + B["tools"][t]))
    print(f"{'tool':<34} {'A calls':>8} {'A tokens':>10} {'B calls':>8} {'B tokens':>10}")
    for t in allt[:18]:
        print(f"{t:<34} {A['tools'][t]:>8} {A['results'][t]:>10} "
              f"{B['tools'][t]:>8} {B['results'][t]:>10}")

    for label, S in (("A", A), ("B", B)):
        ci = sum(v for k, v in S["tools"].items() if "codeindex" in k)
        rd = S["tools"].get("Read", 0)
        print(f"\n{label}: codeindex calls={ci}  Read calls={rd}  "
              f"Read tokens={S['results'].get('Read', 0)}")


if __name__ == "__main__":
    main()
