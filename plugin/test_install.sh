#!/usr/bin/env bash
# Run install.sh against an isolated HOME and check what actually landed.
#
# Every earlier check of this script ran it from inside the repository against
# this machine's own HOME, where the skill directories already existed and the
# last ~/.claude-* sibling happened to be a real Claude home. Three faults hid
# behind that:
#
#   1. `curl … | bash` has no script file, so BASH_SOURCE[0] was unset and
#      `set -u` made referencing it fatal. The documented install printed an
#      unbound-variable error, installed the binaries, skipped the skill, exited 0.
#   2. skill_dirs ended on a filter test. On a machine with only ~/.claude the
#      glob stays literal, that test fails, `pipefail` fails the pipeline and
#      `set -e` killed the script at targets="$(skill_dirs)" — exit 1, no
#      message, no skill, no registration. Every first install on a normal machine.
#   3. The Windows asset was saved without .exe, so the download succeeded and
#      running it could not.
#
# None are visible from the outside: the binary lands either way. So this asserts
# the whole outcome, in both the piped and the checked-out form.
set -uo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(dirname "$here")"
fail=0

stub_dir() {
    d="$1"
    mkdir -p "$d"
    # A release that publishes the current asset names. The content only has to
    # be an executable file: this test checks names, modes and placement, not
    # that the server runs — the MCP handshake is covered elsewhere.
    cat >"$d/gh" <<'EOF'
#!/bin/sh
case "$1 $2" in
    "release download")
        out=""
        while [ $# -gt 0 ]; do
            case "$1" in -O) out="$2"; shift ;; esac
            shift
        done
        printf '#!/bin/sh\necho stub\n' > "$out"
        ;;
    "api "*) printf -- '---\nname: distil\ndescription: stub\n---\nbody\n' ;;
    *) exit 1 ;;
esac
EOF
    # No plugin installed, so the installer owns the skill and the registration.
    cat >"$d/claude" <<'EOF'
#!/bin/sh
case "$1 $2" in
    "plugin list") exit 0 ;;
    "mcp remove") exit 0 ;;
    "mcp add") echo "$*" >> "$MCP_LOG" ;;
    *) exit 0 ;;
esac
EOF
    chmod +x "$d/gh" "$d/claude"
}

check_outcome() {
    mode="$1" home="$2" log="$3" status="$4" out="${5:-}"

    # A clean outcome is not enough. The piped form used to reach the same end
    # state while printing "BASH_SOURCE[0]: unbound variable" first, and an
    # install that opens with an interpreter error is not one anyone trusts.
    if [ -n "$out" ] && [ -f "$out" ] && grep -qE "unbound variable|BASH_SOURCE" "$out"; then
        printf 'FAIL %-10s printed a shell error: %s\n' "$mode" \
            "$(grep -oE '[^ ]*(unbound variable|BASH_SOURCE)[^ ]*' "$out" | head -1)"
        fail=$((fail + 1))
    fi

    if [ "$status" -ne 0 ]; then
        printf 'FAIL %-10s exited %d\n' "$mode" "$status"
        fail=$((fail + 1))
    fi
    for bin in distil-mcp distil-bench; do
        if [ ! -x "$home/.local/bin/$bin" ]; then
            printf 'FAIL %-10s %s missing or not executable\n' "$mode" "$bin"
            fail=$((fail + 1))
        fi
    done
    skill="$home/.claude/skills/distil"
    if [ ! -s "$skill/SKILL.md" ]; then
        printf 'FAIL %-10s skill not installed (%s)\n' "$mode" "$skill/SKILL.md"
        fail=$((fail + 1))
    fi
    if [ ! -f "$skill/.distil-managed" ]; then
        printf 'FAIL %-10s skill has no managed marker, so no later run can update it\n' "$mode"
        fail=$((fail + 1))
    fi
    if ! grep -q "distil" "$log" 2>/dev/null; then
        printf 'FAIL %-10s no MCP registration was attempted\n' "$mode"
        fail=$((fail + 1))
    fi
}

run_case() {
    mode="$1"
    tmp="$(mktemp -d)"
    home="$tmp/home"
    mkdir -p "$home/.claude"
    # One Claude home and nothing else, which is the normal machine the glob
    # fault needed.
    echo '{}' > "$home/.claude/.claude.json"
    stub_dir "$tmp/stub"
    log="$tmp/mcp.log"
    : > "$log"

    if [ "$mode" = "piped" ]; then
        cat "$root/install.sh" | env -u CLAUDE_CONFIG_DIR \
            HOME="$home" MCP_LOG="$log" INSTALL_DIR="$home/.local/bin" \
            PATH="$tmp/stub:/usr/bin:/bin" bash >"$tmp/out" 2>&1
        status=$?
    else
        env -u CLAUDE_CONFIG_DIR \
            HOME="$home" MCP_LOG="$log" INSTALL_DIR="$home/.local/bin" \
            PATH="$tmp/stub:/usr/bin:/bin" bash "$root/install.sh" >"$tmp/out" 2>&1
        status=$?
    fi

    check_outcome "$mode" "$home" "$log" "$status" "$tmp/out"

    # Re-running must be safe: the download refuses an existing path without
    # --clobber, and a skill directory it wrote before carries a marker.
    if [ "$mode" = "checkout" ]; then
        env -u CLAUDE_CONFIG_DIR \
            HOME="$home" MCP_LOG="$log" INSTALL_DIR="$home/.local/bin" \
            PATH="$tmp/stub:/usr/bin:/bin" bash "$root/install.sh" >"$tmp/out2" 2>&1
        status=$?
        check_outcome "rerun" "$home" "$log" "$status" "$tmp/out2"
    fi

    rm -rf "$tmp"
}

run_case piped
run_case checkout

if [ "$fail" -gt 0 ]; then
    printf '\n%d install failure(s)\n' "$fail" >&2
    exit 1
fi
echo "install: piped, checked-out and re-run all land the binaries, the skill and the registration"
