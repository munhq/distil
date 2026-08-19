#!/usr/bin/env python3
"""Does compression destroy the agent's record of what it touched?

Artifact tracking is the weakest measured dimension in this field. Factory.ai
scored every method they tested between 2.19 and 2.45 out of 5.0 on it — worse
than any other probe type, and the scoring was an LLM judging an LLM.

This measures the same thing with ground truth instead of a judge.

A transcript already states, as fact, every file the agent operated on: the
`file_path` argument of each Read, Edit and Write call. That set is not an
opinion. Compress the session, extract the file paths that survive, and the
retention rate is a division, not a grade.

Two failure modes are separated, because they have different consequences:

  DROPPED   a file the agent really touched is no longer mentioned. The model
            can no longer answer "what did we change", and may redo the work.
  ORPHANED  a path survives but the file does not exist on disk. Not caused by
            compression — it is a deleted or renamed file — but it is reported
            so it cannot be mistaken for a compression artifact.
"""
import glob
import json
import os
import re
import sys

FILE_TOOLS = {"Read", "Edit", "Write", "NotebookEdit", "MultiEdit"}
# A path-like token: at least one separator, a plausible extension, no spaces.
PATH_RE = re.compile(r"[\w./~-]*/[\w./-]+\.\w{1,6}")


def truth_set(messages):
    """Files the agent demonstrably operated on, from tool_use arguments."""
    out = set()
    for m in messages:
        if m.get("role") != "assistant":
            continue
        c = m.get("content")
        if not isinstance(c, list):
            continue
        for b in c:
            if not isinstance(b, dict) or b.get("type") != "tool_use":
                continue
            if b.get("name") not in FILE_TOOLS:
                continue
            args = b.get("input") or {}
            p = args.get("file_path") or args.get("path") or args.get("notebook_path")
            if isinstance(p, str) and p:
                out.add(os.path.normpath(p))
    return out


def mentioned_paths(messages):
    """Every path-shaped token anywhere in the serialized messages."""
    blob = json.dumps(messages)
    return {os.path.normpath(p) for p in PATH_RE.findall(blob)}


def main():
    session_dir = sys.argv[1]
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    unlocked = len(sys.argv) > 3

    import headroom

    cfg = headroom.CompressConfig(compress_user_messages=True) if unlocked else None
    print(f"--- artifact retention, Headroom "
          f"{'compress_user_messages=True' if cfg else 'default'} ---")
    print(f"{'session':<14} {'truth':>6} {'kept':>6} {'dropped':>8} "
          f"{'retention':>10} {'orphaned':>9}")

    tot_truth = tot_kept = tot_orphan = 0
    per_session = []
    for f in sorted(glob.glob(f"{session_dir}/*.json"), key=os.path.getsize)[:cap]:
        msgs = json.load(open(f))
        truth = truth_set(msgs)
        if not truth:
            continue
        kw = {"model": "claude-sonnet-4-5-20250929"}
        if cfg:
            kw["config"] = cfg
        try:
            after = headroom.compress(msgs, **kw).messages
        except Exception as e:
            print(f"{os.path.basename(f)[:12]:<14} ERROR {type(e).__name__}")
            continue

        survived = mentioned_paths(after)
        kept = {p for p in truth if p in survived}
        dropped = truth - kept
        orphan = {p for p in truth if not os.path.exists(p)}

        tot_truth += len(truth)
        tot_kept += len(kept)
        tot_orphan += len(orphan)
        per_session.append((len(truth), len(kept), sorted(dropped)[:3]))
        print(f"{os.path.basename(f)[:12]:<14} {len(truth):>6} {len(kept):>6} "
              f"{len(dropped):>8} {len(kept)/len(truth)*100:>9.1f}% {len(orphan):>9}")

    if tot_truth:
        print(f"\nTOTAL artifacts: {tot_truth}   retained: {tot_kept}   "
              f"retention: {tot_kept/tot_truth*100:.1f}%")
        print(f"of the truth set, {tot_orphan} paths no longer exist on disk "
              f"(deleted/renamed, not a compression fault)")
        perfect = sum(1 for t, k, _ in per_session if t == k)
        print(f"sessions with perfect artifact retention: "
              f"{perfect}/{len(per_session)}")
        worst = sorted(per_session, key=lambda r: r[1] / r[0])[:3]
        for t, k, ex in worst:
            if ex:
                print(f"  worst case {k}/{t} kept; dropped e.g. {ex}")


if __name__ == "__main__":
    main()
