#!/usr/bin/env bash
# The shebang matters: this script uses arrays and ${BASH_SOURCE[0]}, neither of
# which exists in every shell. Without it, `./install.sh` runs under whatever
# shell the caller happens to use and fails on the first array.
set -euo pipefail

REPO="${REPO:-munhq/distil}"
INSTALL_DIR="${INSTALL_DIR:-$HOME/.local/bin}"
MARKER=".distil-managed"

# Every Claude home, not only the active one.
#
# Claude Code reads skills from its config directory, and CLAUDE_CONFIG_DIR
# moves that. A machine can hold ~/.claude plus siblings such as ~/.claude-work,
# each with its own skills directory, and installing into one of them looks like
# a success in every account that cannot see the skill — the exact failure the
# skill exists to prevent. Some siblings symlink ~/.claude/skills, so resolve
# each path and drop duplicates rather than copying over the same directory
# several times. An explicit SKILL_DIR overrides all of this.
resolve_dir() {
    if [ -d "$1" ]; then (cd "$1" && pwd -P); else
        parent="$(dirname "$1")"
        [ -d "$parent" ] && printf '%s/%s\n' "$(cd "$parent" && pwd -P)" "$(basename "$1")"
    fi
}

skill_dirs() {
    if [ -n "${SKILL_DIR:-}" ]; then
        resolve_dir "$SKILL_DIR"
        return
    fi
    {
        [ -n "${CLAUDE_CONFIG_DIR:-}" ] && resolve_dir "$CLAUDE_CONFIG_DIR/skills"
        for home in "$HOME"/.claude "$HOME"/.claude-*; do
            # A name glob alone is wrong: ~/.claude-mem, ~/.claude-desktop and
            # ~/.claude-account-backups match it and are not Claude Code homes.
            # Installing there writes files nothing will ever read. Every real
            # home holds .claude.json, so require it.
            [ -f "$home/.claude.json" ] && resolve_dir "$home/skills"
        done
    } | awk 'NF && !seen[$0]++'
}

# Install one skill directory, and never clobber a skill this script did not
# write. Drop-in skills have no native versioning, so each installed directory
# carries a marker naming the version that put it there; a directory without one
# belongs to the user.
install_skill() {
    src="$1" dest_root="$2"
    name="$(basename "$src")"
    target="$dest_root/$name"
    if [ -e "$target" ] && [ ! -f "$target/$MARKER" ]; then
        # No marker, so this directory predates the marker or belongs to the
        # user. Identical content means an earlier run of this script wrote it,
        # and adopting it is a no-op that only adds the marker. Different
        # content is the user's, and overwriting it would be data loss.
        declared="$(sed -n 's/^name:[[:space:]]*//p' "$target/SKILL.md" 2>/dev/null | head -1)"
        if diff -r -q "$src" "$target" >/dev/null 2>&1; then
            echo "adopting existing identical skill at $target" >&2
        elif [ "$declared" = "$name" ]; then
            # Same content is only the unchanged case. An older version of this
            # skill shipped before markers existed and differs by exactly the
            # edits since — refusing it would strand every machine on the copy
            # it happened to install first. A SKILL.md whose frontmatter names
            # this skill is ours; anything else is left alone.
            echo "replacing an older $name skill at $target" >&2
        else
            echo "warning: $target exists, differs from the bundled skill, and" \
                 "carries no marker; left alone" >&2
            return
        fi
    fi
    mkdir -p "$dest_root"
    rm -rf "${target:?}"
    cp -R "$src" "$target"
    printf '%s %s\n' "distil" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$target/$MARKER"
    echo "installed skill -> $target"
}
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

# Map uname onto the names the release publishes. uname disagrees with them on a
# Mac: `uname -s` says Darwin where the asset says macos, and Apple Silicon says
# arm64 where the asset says aarch64. A lowercased uname therefore asked for
# distil-mcp-arm64-darwin, which 404s, so every Apple Silicon Mac fell through to
# a source build behind a message about there being no prebuilt binary — which
# reads as a fact about the release rather than the bug it was.
# plugin/test_platform.sh holds this to the release matrix.
resolve_artifact() {
    local arch os ext=""
    arch="$(uname -m)"
    os="$(uname -s)"
    case "$arch" in
        x86_64|amd64) arch=x86_64 ;;
        arm64|aarch64) arch=aarch64 ;;
        *) return 1 ;;
    esac
    # Git Bash, MSYS2 and Cygwin report a decorated kernel name rather than
    # anything containing "windows", e.g. MINGW64_NT-10.0-22631.
    case "$os" in
        Linux) os=linux ;;
        Darwin) os=macos ;;
        MINGW*|MSYS*|CYGWIN*|Windows_NT) os=windows; ext=".exe" ;;
        *) return 1 ;;
    esac
    printf '%s-%s-%s%s\n' "$1" "$arch" "$os" "$ext"
}

# Introspection for plugin/test_platform.sh, before anything is installed.
if [ "${1:-}" = "--print-artifact" ]; then
    for entry in "${BINARIES[@]}"; do
        resolve_artifact "${entry%%:*}" || { echo "unsupported"; exit 1; }
    done
    exit 0
fi

need_build=()
for entry in "${BINARIES[@]}"; do
    bin="${entry%%:*}"
    if ! artifact="$(resolve_artifact "$bin")"; then
        echo "no release build for $(uname -m)-$(uname -s); building from source" >&2
        need_build+=("$entry")
        continue
    fi
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
    echo "no prebuilt binary for this platform; building from source"
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
    targets="$(skill_dirs)"
    if [ -z "$targets" ]; then
        echo "warning: no Claude skills directory found; skills not installed" >&2
    else
        for dest in $targets; do
            for skill in "$SRC_DIR"/plugin/skills/*/; do
                install_skill "${skill%/}" "$dest"
            done
        done
    fi
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
echo "  skills : $(skill_dirs | tr '\n' ' ')"
case ":$PATH:" in
    *":$INSTALL_DIR:"*) ;;
    *) echo; echo "note: $INSTALL_DIR is not on your PATH." ;;
esac
