#!/usr/bin/env bash
#
# Install distil: the MCP server, the measurement CLI, and the Agent Skills.
#
# The skills matter as much as the binaries. Measured on one machine over 13,694
# sessions, an MCP server that ships skills was actually called in 18.5% of the
# sessions where it was available; one without them managed 4.1%. Registering a
# server without telling an agent when to reach for it mostly produces an idle
# server, so this script installs both or reports which half it could not.
set -euo pipefail

REPO="${REPO:-munhq/distil}"
INSTALL_DIR="${INSTALL_DIR:-$HOME/.local/bin}"
SKILL_DIR="${SKILL_DIR:-$HOME/.claude/skills}"
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# distil-mcp is the server an agent talks to; distil-bench is the CLI a person
# runs. Both are feature-gated, and omitting the feature is why an earlier
# version of this script could not build either from source.
BINARIES=("distil-mcp:mcp" "distil-bench:bench")

mkdir -p "$INSTALL_DIR"

ARCH="$(uname -m)"
OS="$(uname -s | tr '[:upper:]' '[:lower:]')"

need_source_build=0
for entry in "${BINARIES[@]}"; do
    bin="${entry%%:*}"
    artifact="${bin}-${ARCH}-${OS}"
    if command -v gh &>/dev/null &&
       gh release download --repo "$REPO" -p "$artifact" -O "$INSTALL_DIR/$bin" 2>/dev/null; then
        chmod +x "$INSTALL_DIR/$bin"
        echo "installed prebuilt $bin -> $INSTALL_DIR/$bin"
    else
        need_source_build=1
    fi
done

if [ "$need_source_build" -eq 1 ]; then
    echo "no prebuilt binary for ${ARCH}-${OS}; building from source"
    if ! command -v cargo &>/dev/null; then
        echo "error: cargo not found. Install Rust from https://rustup.rs" >&2
        exit 1
    fi
    for entry in "${BINARIES[@]}"; do
        bin="${entry%%:*}"
        feat="${entry##*:}"
        [ -x "$INSTALL_DIR/$bin" ] && continue
        # Both binaries are behind required-features, so the feature flag is
        # mandatory: without it cargo reports no such binary.
        ( cd "$SRC_DIR" && cargo build --release --features "$feat" --bin "$bin" )
        cp "$SRC_DIR/target/release/$bin" "$INSTALL_DIR/$bin"
        echo "built and installed $bin -> $INSTALL_DIR/$bin"
    done
fi

# ── Skills ──────────────────────────────────────────────────────────────────
if [ -d "$SRC_DIR/plugin/skills" ]; then
    mkdir -p "$SKILL_DIR"
    for skill in "$SRC_DIR"/plugin/skills/*/; do
        name="$(basename "$skill")"
        rm -rf "${SKILL_DIR:?}/$name"
        cp -R "$skill" "$SKILL_DIR/$name"
        echo "installed skill -> $SKILL_DIR/$name"
    done
else
    echo "warning: no plugin/skills directory found; skills not installed" >&2
fi

# ── MCP registration ────────────────────────────────────────────────────────
if command -v claude &>/dev/null; then
    # Re-registering the same name errors rather than replacing, so drop any
    # previous entry first and keep the script re-runnable.
    claude mcp remove distil 2>/dev/null || true
    claude mcp add distil "$INSTALL_DIR/distil-mcp"
    echo "registered distil with Claude Code"
else
    echo "Claude Code not found — register manually:"
    echo "  claude mcp add distil $INSTALL_DIR/distil-mcp"
fi

echo
echo "done."
echo "  server : $INSTALL_DIR/distil-mcp"
echo "  cli    : $INSTALL_DIR/distil-bench ~/.claude/projects"
echo "  skills : $SKILL_DIR"
case ":$PATH:" in
    *":$INSTALL_DIR:"*) ;;
    *) echo; echo "note: $INSTALL_DIR is not on your PATH." ;;
esac
