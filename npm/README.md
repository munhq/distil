<img src="https://raw.githubusercontent.com/munhq/distil/main/docs/brand/logo.svg" alt="distil" width="210" height="70">

Context optimization middleware for LLM agents — a dynamic tool registry, result masking, token budgeting and smart compaction, so a long session stays inside its window instead of failing for length.

```
npx -y @munhq/distil
```

No account, no API key, no configuration.

## Add it to a client

Claude Code:

```
claude mcp add distil -- npx -y @munhq/distil
```

Anything that reads a JSON config (Claude Desktop, Cursor, Windsurf, Zed, Cline):

```json
{
  "mcpServers": {
    "distil": {
      "command": "npx",
      "args": [
        "-y",
        "@munhq/distil"
      ]
    }
  }
}
```

## Why this is an npm package when the server is not JavaScript

distil is a compiled binary. This package is a small wrapper: on install it resolves the release asset for your platform, verifies it against the `checksums.txt` published beside it, caches it under `~/.cache/distil/bin/distil-<version>`, and executes it. `DISTIL_BIN` overrides everything, for a local build; `PATH` is deliberately not searched, because this package declares one version to the MCP registry.

Source, the other install paths and the full tool list: **https://github.com/munhq/distil**
