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
        # These tests filter candidates; a false one is not a failure. The
        # block's status is the last test's, and on a machine with only ~/.claude
        # the glob "$HOME"/.claude-* stays literal, so that last test fails.
        # `set -o pipefail` then failed the whole pipeline and `set -e` killed
        # the script at targets="$(skill_dirs)" — the installer exited 1 without
        # a message, after the binaries and before the skill and the MCP
        # registration. On this machine the last sibling happened to be a real
        # home, so it passed here and would have failed for everyone else.
        true
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
# Empty when piped. `curl … | bash` has no script file, so BASH_SOURCE[0] is
# unset and `set -u` made referencing it fatal: the piped install printed
# "BASH_SOURCE[0]: unbound variable", installed the binaries, silently skipped
# the skill, and still exited 0. Every earlier test ran `bash install.sh` from
# inside the repo, where the variable is set and plugin/skills is right there,
# so the bug was only reachable through the piped form.
if [ -n "${BASH_SOURCE[0]:-}" ] && [ -f "${BASH_SOURCE[0]}" ]; then
    SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
else
    SRC_DIR=""
fi

BINARIES=("distil-mcp:mcp" "distil-bench:bench")

# Windows will not execute a file without the .exe extension, so the installed
# name carries it. The download would have succeeded and running it would have
# failed. Empty everywhere else, so nothing changes on Linux or macOS.
case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*|Windows_NT) EXE=".exe" ;;
    *) EXE="" ;;
esac

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
        atomic_install "$dl" "$bin$EXE"
        rm -f "$dl"
        echo "installed prebuilt $bin -> $INSTALL_DIR/$bin$EXE"
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
    # A piped install has no checkout, so there is nothing to build. Saying so
    # beats running cargo in whatever directory the pipe happened to start in.
    if [ -z "$SRC_DIR" ]; then
        echo "error: no local checkout to build from. Clone the repository:" >&2
        echo "  git clone git@github.com:$REPO.git && cd distil && ./install.sh" >&2
        exit 1
    fi
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
        atomic_install "$SRC_DIR/target/release/$bin$EXE" "$bin$EXE"
        echo "built and installed $bin -> $INSTALL_DIR/$bin$EXE"
    done
fi

# The plugin channel ships the skill and declares the MCP server itself. When it
# is installed, doing both again leaves two definitions of one server and two
# copies of one skill, so the binary is all this script contributes.
plugin_installed() {
    command -v claude >/dev/null 2>&1 || return 1
    claude plugin list 2>/dev/null | grep -q "distil@"
}

if plugin_installed; then
    PLUGIN_OWNS=1
    echo "distil plugin detected — it provides the skill and the MCP server."
    echo "installed the binary only, so the plugin launcher resolves it locally."
else
    PLUGIN_OWNS=0
fi

if [ "$PLUGIN_OWNS" = "1" ]; then
    echo "skipping skill install: the plugin ships it."
elif [ -d "$SRC_DIR/plugin/skills" ]; then
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
    # Piped install: no checkout to copy from, so fetch the skill from the same
    # repository. One file per skill, so this is a download rather than a clone.
    # gh first, because this repository is private and plain curl cannot read it.
    targets="$(skill_dirs)"
    if [ -z "$targets" ]; then
        echo "warning: no Claude skills directory found; skills not installed" >&2
    else
        STAGE="$(mktemp -d)"
        trap 'rm -rf "$STAGE"' EXIT
        fetched=0
        for name in distil; do
            mkdir -p "$STAGE/$name"
            path="plugin/skills/$name/SKILL.md"
            if command -v gh &>/dev/null &&
               gh api "repos/$REPO/contents/$path" -H "Accept: application/vnd.github.raw" \
                  > "$STAGE/$name/SKILL.md" 2>/dev/null &&
               [ -s "$STAGE/$name/SKILL.md" ]; then
                fetched=$((fetched + 1))
            elif command -v curl &>/dev/null &&
                 curl -fsSL -o "$STAGE/$name/SKILL.md" \
                   "https://raw.githubusercontent.com/$REPO/main/$path"; then
                fetched=$((fetched + 1))
            else
                echo "warning: could not fetch the $name skill" >&2
                rm -rf "$STAGE/$name"
            fi
        done
        if [ "$fetched" -gt 0 ]; then
            for dest in $targets; do
                for skill in "$STAGE"/*/; do
                    install_skill "${skill%/}" "$dest"
                done
            done
        fi
    fi
fi

if [ "$PLUGIN_OWNS" = "1" ]; then
    echo "skipping MCP registration: the plugin declares the server."
elif command -v claude &>/dev/null; then
    # -s user, because `claude mcp add` defaults to local scope and would
    # register the server for this one directory only. Re-adding a name that
    # exists errors instead of replacing it, so remove first and keep the
    # script re-runnable.
    claude mcp remove -s user distil 2>/dev/null || true
    claude mcp remove distil 2>/dev/null || true
    claude mcp add -s user distil "$INSTALL_DIR/distil-mcp$EXE"
    echo "registered distil with Claude Code (user scope)"
else
    echo "Claude Code not found — register manually:"
    echo "  claude mcp add -s user distil $INSTALL_DIR/distil-mcp$EXE"
fi

echo
echo "done."
echo "  server : $INSTALL_DIR/distil-mcp$EXE"
echo "  cli    : $INSTALL_DIR/distil-bench$EXE ~/.claude/projects"
echo "  skills : $(skill_dirs | tr '\n' ' ')"
case ":$PATH:" in
    *":$INSTALL_DIR:"*) ;;
    *) echo; echo "note: $INSTALL_DIR is not on your PATH." ;;
esac
