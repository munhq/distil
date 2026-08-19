"""Per-session Headroom run with progress, so a slow session cannot hide the rest.

Prints one line per session as it completes. Pass any second argument to run with
compress_user_messages=True, which makes tool results eligible — in the Anthropic
wire format a tool result is carried inside a user message, so the default config
excludes exactly the content that dominates a coding session.
"""
import glob
import json
import os
import sys
import time

import headroom
import tiktoken

ENC = tiktoken.get_encoding("cl100k_base")


def nt(m):
    return len(ENC.encode(json.dumps(m), disallowed_special=()))


def first_mod(a, b):
    """Index of the first differing message — where the cached prefix dies."""
    for i in range(min(len(a), len(b))):
        if json.dumps(a[i], sort_keys=True) != json.dumps(b[i], sort_keys=True):
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


cfg = headroom.CompressConfig(compress_user_messages=True) if len(sys.argv) > 1 else None
label = "compress_user_messages=True" if cfg else "DEFAULT config"
print(f"--- Headroom 0.35.0, {label} ---", flush=True)

rows = []
for f in sorted(glob.glob("sessions_small/*.json"), key=os.path.getsize):
    m = json.load(open(f))
    b = nt(m)
    t0 = time.time()
    try:
        kw = {"model": "claude-sonnet-4-5-20250929"}
        if cfg:
            kw["config"] = cfg
        out = headroom.compress(m, **kw)
        a = nt(out.messages)
        idx = first_mod(m, out.messages)
    except Exception as e:
        print(f"{os.path.basename(f)[:12]} msgs={len(m):<5} ERROR {type(e).__name__}", flush=True)
        continue
    el = time.time() - t0
    kept = nt(m[:idx]) if idx is not None else b
    rows.append((b, a, kept, idx, len(m)))
    print(
        f"{os.path.basename(f)[:12]} msgs={len(m):<5} {b:>8}->{a:<8} "
        f"saved={(b - a) / b * 100:5.1f}%  first_edit={str(idx):<6} {el:5.1f}s",
        flush=True,
    )

if rows:
    tb = sum(r[0] for r in rows)
    ta = sum(r[1] for r in rows)
    ch = [r for r in rows if r[3] is not None]
    print(f"\nTOTAL {tb:,} -> {ta:,}  saved {(tb - ta) / tb * 100:.1f}%   "
          f"modified {len(ch)}/{len(rows)} sessions")
    if ch:
        kept = sum(r[2] for r in ch)
        tot = sum(r[0] for r in ch)
        ib = sum(r[0] - r[2] for r in ch)
        ia = sum(r[1] - r[2] for r in ch)
        print(f"cached prefix surviving: {kept / tot * 100:.1f}% of tokens")
        if ib > 0:
            print(f"invalidated tail shrank to {ia / ib * 100:.1f}% of its size")
            for k in (10, 20, 50):
                need = 0.1 * k / (1.25 - 0.1 + 0.1 * k)
                verdict = "PAYS" if ia / ib < need else "LOSES"
                print(f"  {k:>3} turns remaining: break-even {need * 100:.1f}% -> {verdict}")
        fr = sorted(r[3] / r[4] for r in ch)
        print(f"first edit at median {fr[len(fr) // 2] * 100:.0f}% through the conversation")
