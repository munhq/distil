#!/usr/bin/env python3
"""Does codeindex answer a Read at lower token cost? Measured per call.

The session-level A/B was hopelessly confounded: only 28 sessions ever called a
codeindex tool, and all of them were long, so session length produced the entire
apparent effect. This measures the same question causally instead.

For each REAL `Read` call recorded in the corpus whose file still exists, ask
codeindex the structural equivalent of the same request and compare token cost:

    Read(path)                  -> the whole file enters context
    get_outline(path)           -> symbols and line counts only
    read_symbol(path, symbol)   -> one symbol's source

This is a fair comparison only where the questions match. `Read` and
`get_outline` do NOT answer the same question in general: if the agent needed a
specific line, an outline cannot substitute. So the outcome reported is a
CEILING on what codeindex could save, not a claim that every Read was wasteful.
The share of Read calls whose result the agent then quoted or edited is the part
that would need a different study.

Speaks MCP JSON-RPC over stdio, the same interface an agent uses.
"""
import json
import os
import subprocess
import sys
import time
from collections import defaultdict

import tiktoken

ENC = tiktoken.get_encoding("cl100k_base")


def ntok(s):
    return len(ENC.encode(s or "", disallowed_special=()))


class CodeIndex:
    """Minimal MCP stdio client."""

    def __init__(self, binary, workspace):
        self.p = subprocess.Popen(
            [binary, "--mcp", "--workspace", workspace],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1,
        )
        self.id = 0
        self._rpc("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "distil-bench", "version": "0"},
        })
        self._notify("notifications/initialized", {})

    def _send(self, obj):
        self.p.stdin.write(json.dumps(obj) + "\n")
        self.p.stdin.flush()

    def _notify(self, method, params):
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def _rpc(self, method, params, timeout=60):
        self.id += 1
        mid = self.id
        self._send({"jsonrpc": "2.0", "id": mid, "method": method, "params": params})
        deadline = time.time() + timeout
        while time.time() < deadline:
            line = self.p.stdout.readline()
            if not line:
                return None
            try:
                msg = json.loads(line)
            except Exception:
                continue
            if msg.get("id") == mid:
                return msg
        return None

    def call(self, tool, args):
        r = self._rpc("tools/call", {"name": tool, "arguments": args})
        if not r or "result" not in r:
            return None
        parts = []
        for c in r["result"].get("content", []):
            if c.get("type") == "text":
                parts.append(c.get("text", ""))
        return "\n".join(parts)

    def close(self):
        try:
            self.p.terminate()
            self.p.wait(timeout=5)
        except Exception:
            self.p.kill()


def main():
    payloads, binary = sys.argv[1], sys.argv[2]
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 200

    # Group surviving Read calls by workspace, since codeindex indexes a root.
    by_ws = defaultdict(list)
    for line in open(payloads):
        d = json.loads(line)
        if d["tool"] != "Read":
            continue
        try:
            a = json.loads(d.get("args") or "{}")
        except Exception:
            continue
        path = a.get("file_path") or a.get("path")
        if not path or not os.path.isfile(path):
            continue
        # Walk up to a repository root; codeindex refuses to index a home dir.
        ws, cur = None, os.path.dirname(os.path.abspath(path))
        while cur and cur != "/":
            if os.path.isdir(os.path.join(cur, ".git")):
                ws = cur
                break
            cur = os.path.dirname(cur)
        if ws:
            by_ws[ws].append((path, d["tokens"]))

    total = sum(len(v) for v in by_ws.values())
    print(f"resolvable Read calls: {total} across {len(by_ws)} repos")

    rows = []
    done = 0
    for ws, items in sorted(by_ws.items(), key=lambda kv: -len(kv[1])):
        if done >= limit:
            break
        try:
            ci = CodeIndex(binary, ws)
        except Exception as e:
            print(f"  skip {ws}: {type(e).__name__}")
            continue
        # Indexing runs in the background; give it a moment to settle.
        ci.call("index_workspace", {"workspace": ws})
        for path, read_tokens in items:
            if done >= limit:
                break
            rel = os.path.relpath(path, ws)
            # The schema names this `path`. An unknown or unindexed file comes
            # back as the human string "No outline found for: ...", NOT as an
            # error object — so a None check alone silently scores failures as
            # a 6-token win. Validate the shape instead.
            out = ci.call("get_outline", {"path": rel})
            if not out or not out.lstrip().startswith("{"):
                continue
            try:
                doc = json.loads(out)
            except Exception:
                continue
            if not doc.get("symbols"):
                continue
            rows.append({
                "ws": ws, "path": rel,
                "read": read_tokens,
                "outline": ntok(out),
                "symbols": len(doc.get("symbols", [])),
                "lines": doc.get("line_count", 0),
            })
            done += 1
        ci.close()

    ok = [r for r in rows if r["outline"] > 0]
    print(f"compared {len(ok)} files")
    if not ok:
        print("no outlines returned — check the tool's argument name")
        return

    tr = sum(r["read"] for r in ok)
    to = sum(r["outline"] for r in ok)
    print(f"\nRead total    {tr:>10,} tokens  ({tr / len(ok):>8.0f} per call)")
    print(f"outline total {to:>10,} tokens  ({to / len(ok):>8.0f} per call)")
    print(f"outline costs {to / tr * 100:.1f}% of Read  ->  {(1 - to / tr) * 100:.1f}% cheaper")

    ratios = sorted(r["outline"] / r["read"] for r in ok if r["read"] > 0)
    print(f"per-file ratio: p10 {ratios[len(ratios)//10]*100:.0f}%  "
          f"p50 {ratios[len(ratios)//2]*100:.0f}%  "
          f"p90 {ratios[len(ratios)*9//10]*100:.0f}%")
    worse = sum(1 for x in ratios if x >= 1.0)
    print(f"files where the outline is NOT smaller: {worse}/{len(ratios)}")

    big = sorted(ok, key=lambda r: -r["read"])[:8]
    print(f"\n{'file':<52} {'Read':>8} {'outline':>8} {'ratio':>7}")
    for r in big:
        print(f"{r['path'][:52]:<52} {r['read']:>8} {r['outline']:>8} "
              f"{r['outline']/r['read']*100:>6.0f}%")


if __name__ == "__main__":
    main()
