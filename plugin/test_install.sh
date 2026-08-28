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

# A stub release on disk. install.sh reads it through file:// URLs, so the real
# http_get and the real checksum verification both run without touching the
# network. A stub that skipped checksums.txt would leave the one step that
# refuses a tampered binary untested.
stub_release() {
    d="$1"
    mkdir -p "$d"
    for a in $(bash "$root/install.sh" --print-artifact); do
        printf '#!/bin/sh\necho stub\n' > "$d/$a"
    done
    ( cd "$d" && sha256sum ./* | sed 's| \./| |' > checksums.txt )
}

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
    stub_release "$tmp/release"
    if [ "$mode" = "gh-fallback" ]; then
        # No working HTTP client. `command -v` finds these first, so install.sh
        # takes them and they fail — the situation the gh path exists for.
        for c in curl wget; do
            printf '#!/bin/sh\nexit 1\n' > "$tmp/stub/$c"; chmod +x "$tmp/stub/$c"
        done
    fi
    log="$tmp/mcp.log"
    : > "$log"

    env_common=(-u CLAUDE_CONFIG_DIR
        HOME="$home" MCP_LOG="$log" INSTALL_DIR="$home/.local/bin"
        RELEASE_URL="file://$tmp/release" RAW_URL="file://$root"
        PATH="$tmp/stub:/usr/bin:/bin")

    if [ "$mode" = "piped" ]; then
        cat "$root/install.sh" | env "${env_common[@]}" bash >"$tmp/out" 2>&1
        status=$?
    else
        env "${env_common[@]}" bash "$root/install.sh" >"$tmp/out" 2>&1
        status=$?
    fi

    check_outcome "$mode" "$home" "$log" "$status" "$tmp/out"

    # WHICH path installed the binary, not just that one landed. install.sh used
    # to reach the release only through gh, so a machine without the CLI — or
    # with it but logged out — fell through to a source build behind a message
    # about there being no prebuilt binary. Nothing failed; the wrong path just
    # became the only one. Assert the path.
    if [ "$mode" = "gh-fallback" ]; then
        if ! grep -q "via gh" "$tmp/out"; then
            printf 'FAIL %-11s did not fall back to gh with no HTTP client\n' "$mode"
            fail=$((fail + 1))
        fi
    elif ! grep -q "checksum verified" "$tmp/out"; then
        printf 'FAIL %-11s did not install over verified HTTP\n' "$mode"
        fail=$((fail + 1))
    fi

    # Re-running must be safe: the download refuses an existing path without
    # --clobber, and a skill directory it wrote before carries a marker.
    if [ "$mode" = "checkout" ]; then
        env "${env_common[@]}" bash "$root/install.sh" >"$tmp/out2" 2>&1
        status=$?
        check_outcome "rerun" "$home" "$log" "$status" "$tmp/out2"

        # A tampered asset must be refused rather than installed. Publishing
        # checksums.txt beside the binaries buys nothing if nothing checks it.
        first="$(bash "$root/install.sh" --print-artifact | head -1)"
        printf 'tampered\n' > "$tmp/release/$first"
        env "${env_common[@]}" bash "$root/install.sh" >"$tmp/out3" 2>&1 || true
        if ! grep -q "checksum mismatch" "$tmp/out3"; then
            printf 'FAIL %-11s a tampered release asset was not rejected\n' tamper
            fail=$((fail + 1))
        fi
    fi

    rm -rf "$tmp"
}

run_case piped
run_case checkout
run_case gh-fallback

if [ "$fail" -gt 0 ]; then
    printf '\n%d install failure(s)\n' "$fail" >&2
    exit 1
fi
echo "install: piped, checked-out, re-run and gh-fallback all land the binaries, the skill and the registration; a tampered asset is refused"
