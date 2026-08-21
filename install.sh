#!/usr/bin/env bash
# The shebang matters: this script uses arrays and ${BASH_SOURCE[0]}, neither of
# which exists in every shell. Without it, `./install.sh` runs under whatever
# shell the caller happens to use and fails on the first array.
set -euo pipefail

REPO="${REPO:-munhq/distil}"
INSTALL_DIR="${INSTALL_DIR:-$HOME/.local/bin}"
# Claude Code reads skills from its config directory, and that is not always
# ~/.claude: CLAUDE_CONFIG_DIR moves it, and this machine runs several accounts
# whose skills directories are separate. A fixed $HOME/.claude/skills installed
# the skill where the running account could not see it, so nothing ever routed
# an agent to this server — the exact failure the skill exists to prevent.
SKILL_DIR="${SKILL_DIR:-${CLAUDE_CONFIG_DIR:-$HOME/.claude}/skills}"
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BINARIES=("distil-mcp:mcp" "distil-bench:bench")

mkdir -p "$INSTALL_DIR"

# Install file $1 as $INSTALL_DIR/$2 atomically. distil-mcp is a long-lived
# server, so a client can hold the target mapped while this runs. Writing it in
# place can SIGBUS that process when it faults in a page of a half-written file.
# Stage on the same filesystem and rename() over the target: the running
# instance keeps the old inode until it exits.
atomic_install() {
    local src="$1" dest="$INSTALL_DIR/$2" tmp
    tmp="$(mktemp "$dest.XXXXXX")"
    cat "$src" > "$tmp"
    chmod 0755 "$tmp"
    mv -f "$tmp" "$dest"
}

ARCH="$(uname -m)"
OS="$(uname -s | tr '[:upper:]' '[:lower:]')"

need_build=()
for entry in "${BINARIES[@]}"; do
    bin="${entry%%:*}"
    artifact="${bin}-${ARCH}-${OS}"
    dl="$INSTALL_DIR/.${bin}.download"
    rm -f "$dl"
    # --clobber is required, not defensive: gh refuses -O onto a path that
    # already exists. Without it the second run of this script downloads
    # nothing, reports no error, and keeps whatever binary is already there.
    if command -v gh &>/dev/null &&
       gh release download --repo "$REPO" -p "$artifact" -O "$dl" --clobber 2>/dev/null; then
        atomic_install "$dl" "$bin"
        rm -f "$dl"
        echo "installed prebuilt $bin -> $INSTALL_DIR/$bin"
    else
        rm -f "$dl"
        need_build+=("$entry")
    fi
done

# Build only what the download did not supply. An earlier version skipped the
# build when the target file existed, which meant a stale binary was never
# replaced — the download failed and the build declined, so the script became a
# silent no-op.
if [ "${#need_build[@]}" -gt 0 ]; then
    echo "no prebuilt binary for ${ARCH}-${OS}; building from source"
    if ! command -v cargo &>/dev/null; then
        echo "error: cargo not found. Install Rust from https://rustup.rs" >&2
        exit 1
    fi
    for entry in "${need_build[@]}"; do
        bin="${entry%%:*}"
        feat="${entry##*:}"
        # Both binaries are behind required-features, so a build without the
        # feature produces nothing at all.
        ( cd "$SRC_DIR" && cargo build --release --features "$feat" --bin "$bin" )
        atomic_install "$SRC_DIR/target/release/$bin" "$bin"
        echo "built and installed $bin -> $INSTALL_DIR/$bin"
    done
fi

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

if command -v claude &>/dev/null; then
    # -s user, because `claude mcp add` defaults to local scope and would
    # register the server for this one directory only. Re-adding a name that
    # exists errors instead of replacing it, so remove first and keep the
    # script re-runnable.
    claude mcp remove -s user distil 2>/dev/null || true
    claude mcp remove distil 2>/dev/null || true
    claude mcp add -s user distil "$INSTALL_DIR/distil-mcp"
    echo "registered distil with Claude Code (user scope)"
else
    echo "Claude Code not found — register manually:"
    echo "  claude mcp add -s user distil $INSTALL_DIR/distil-mcp"
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
