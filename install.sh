#!/usr/bin/env bash
set -euo pipefail

BINARY="distil-mcp"
INSTALL_DIR="${INSTALL_DIR:-$HOME/.local/bin}"
REPO="munhq/distil"

mkdir -p "$INSTALL_DIR"

ARCH="$(uname -m)"
OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
ARTIFACT="${BINARY%-mcp}-${ARCH}-${OS}"

if gh release download v0.1.0 --repo "$REPO" -p "$ARTIFACT" -O "$INSTALL_DIR/$BINARY" 2>/dev/null; then
    chmod +x "$INSTALL_DIR/$BINARY"
    echo "Installed prebuilt binary to $INSTALL_DIR/$BINARY"
else
    echo "No prebuilt binary for $ARCH-$OS, building from source..."
    if ! command -v cargo &>/dev/null; then
        echo "Error: cargo not found. Install Rust: https://rustup.rs" >&2
        exit 1
    fi
    cargo build --release --bin "$BINARY"
    cp "target/release/$BINARY" "$INSTALL_DIR/$BINARY"
    echo "Built and installed to $INSTALL_DIR/$BINARY"
fi

if command -v claude &>/dev/null; then
    claude mcp add distil "$INSTALL_DIR/$BINARY"
    echo "Registered with Claude Code"
else
    echo "Claude Code not found — register manually:"
    echo "  claude mcp add distil $INSTALL_DIR/$BINARY"
fi
