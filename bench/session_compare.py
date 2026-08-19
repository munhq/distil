#!/usr/bin/env python3
"""Compare context compressors on WHOLE real agent sessions.

The earlier payload-level harness was wrong and its numbers were discarded.
Headroom protects the last four messages, skips anything under 250 tokens, and
by default does not touch user messages — and in the Anthropic wire format tool
results ARE user messages. Feeding it one payload measured the harness.

This runs each compressor over complete sessions taken from real transcripts,
under two configurations:

  default   what a user gets with no configuration
  unlocked  compress_user_messages=True, so tool results become eligible

Both are reported. Presenting only the unlocked figure would overstate what the
tool does out of the box; presenting only the default would understate what it
can do.

It also measures something no published benchmark reports: the index of the
FIRST message a compressor modifies. Prompt caching bills a matched prefix at
0.1x, so an edit at message 5 of 200 invalidates everything after it and turns
those reads into writes at 1.25x-2.0x. A compressor that saves 30% by rewriting
the head of the conversation can cost more than it saves.
"""
import glob
import json
import sys
import time
from collections import defaultdict

import tiktoken

ENC = tiktoken.get_encoding("cl100k_base")


def ntok_msgs(msgs) -> int:
    return len(ENC.encode(json.dumps(msgs), disallowed_special=()))


def ntok(s: str) -> int:
    return len(ENC.encode(s, disallowed_special=()))


def first_modified_index(before, after):
    """Index of the first message that differs, or None when nothing changed.

    Compared on serialized form so a change anywhere inside a block counts.
    A length change alone also counts as a modification at the first index
    where the arrays diverge.
    """
    n = min(len(before), len(after))
    for i in range(n):
        if json.dumps(before[i], sort_keys=True) != json.dumps(after[i], sort_keys=True):
            return i
    if len(before) != len(after):
        return n
    return None


def prefix_tokens(msgs, upto):
    """Tokens in messages[0:upto] — the part of the cache that survives."""
    if upto <= 0:
        return 0
    return ntok_msgs(msgs[:upto])


def break_even_fraction(remaining_turns, write_mult=1.25):
    """Max surviving fraction of the invalidated tail for a rewrite to pay.

    Keeping the cache costs K * N * 0.1. Rewriting costs one write at
    `write_mult` plus K-1 later reads at 0.1. Solving for M/N gives:
        0.1 * K / (write_mult - 0.1 + 0.1 * K)
    """
    return 0.1 * remaining_turns / (write_mult - 0.1 + 0.1 * remaining_turns)


def run(sessions, label, config=None):
    import headroom

    rows = []
    for path, msgs in sessions:
        before_tok = ntok_msgs(msgs)
        t0 = time.time()
        try:
            kw = {"model": "claude-sonnet-4-5-20250929"}
            if config is not None:
                kw["config"] = config
            out = headroom.compress(msgs, **kw)
            after = out.messages
        except Exception as e:
            rows.append({
                "path": path, "before": before_tok, "after": None,
                "err": f"{type(e).__name__}: {e}",
            })
            continue
        after_tok = ntok_msgs(after)
        idx = first_modified_index(msgs, after)
        rows.append({
            "path": path,
            "before": before_tok,
            "after": after_tok,
            "n_msgs": len(msgs),
            "first_mod": idx,
            "kept_prefix": prefix_tokens(msgs, idx) if idx is not None else before_tok,
            "secs": time.time() - t0,
            "transforms": list(out.transforms_applied)[:6],
        })
    report(rows, label)
    return rows


def report(rows, label):
    ok = [r for r in rows if r.get("after") is not None]
    errs = [r for r in rows if r.get("after") is None]
    tb = sum(r["before"] for r in ok)
    ta = sum(r["after"] for r in ok)
    print(f"\n=== {label} ===")
    print(f"sessions: {len(ok)} ok, {len(errs)} failed")
    if not ok:
        for e in errs[:3]:
            print("  ", e["err"])
        return
    print(f"tokens: {tb:,} -> {ta:,}   saved {(tb - ta) / tb * 100:.1f}%")

    changed = [r for r in ok if r["first_mod"] is not None]
    print(f"sessions actually modified: {len(changed)}/{len(ok)}")
    if changed:
        # Where the edit lands decides the cache bill.
        fracs = sorted(r["first_mod"] / r["n_msgs"] for r in changed)
        med = fracs[len(fracs) // 2]
        print(f"first edit lands at median {med * 100:.0f}% through the conversation")
        kept = sum(r["kept_prefix"] for r in changed)
        tot = sum(r["before"] for r in changed)
        print(f"cached prefix that survives: {kept / tot * 100:.1f}% of tokens")

        # Of the tail it invalidated, how much did it actually remove?
        inval_before = sum(r["before"] - r["kept_prefix"] for r in changed)
        inval_after = sum(r["after"] - r["kept_prefix"] for r in changed)
        if inval_before > 0:
            surviving = inval_after / inval_before
            print(f"invalidated tail shrank to {surviving * 100:.1f}% of its size")
            for k in (10, 20, 50):
                need = break_even_fraction(k)
                verdict = "PAYS" if surviving < need else "LOSES"
                print(f"  with {k:>3} turns remaining, break-even is "
                      f"{need * 100:.1f}%  -> {verdict}")

    tf = defaultdict(int)
    for r in ok:
        for t in r["transforms"]:
            tf[t.split(":")[0] if ":" in t else t] += 1
    print("transform families:", dict(sorted(tf.items(), key=lambda kv: -kv[1])[:6]))
    print(f"median wall time: {sorted(r['secs'] for r in ok)[len(ok) // 2]:.2f}s")


def main():
    d = sys.argv[1]
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    files = sorted(glob.glob(f"{d}/*.json"))[:cap]
    sessions = []
    for f in files:
        try:
            m = json.load(open(f))
        except Exception:
            continue
        if isinstance(m, list) and m:
            sessions.append((f, m))
    print(f"loaded {len(sessions)} sessions")

    import headroom

    run(sessions, "Headroom 0.35.0 — DEFAULT config")

    cfg = headroom.CompressConfig(compress_user_messages=True)
    run(sessions, "Headroom 0.35.0 — compress_user_messages=True", config=cfg)


if __name__ == "__main__":
    main()
