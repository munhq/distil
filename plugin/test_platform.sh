#!/usr/bin/env bash
# Check that install.sh and the plugin launcher ask for asset names the release
# actually publishes, on every platform the release builds for.
#
# They did not. Both derived the name from uname, which disagrees with the
# release on a Mac: `uname -s` says Darwin where the asset says macos, and Apple
# Silicon says arm64 where the asset says aarch64. So every Apple Silicon Mac
# asked for distil-mcp-arm64-darwin, got a 404, and fell back to needing a Rust
# toolchain — behind a message about there being no prebuilt binary, which reads
# as a fact about the release rather than the bug it was.
#
# The matrix in .github/workflows/release.yml is the source of truth. Add a
# target there without teaching the scripts about it and this fails, rather than
# a stranger's install quietly falling back to a source build.
set -uo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(dirname "$here")"
workflow="$root/.github/workflows/release.yml"
binaries=(distil-mcp distil-bench)

# Every artifact the release publishes, composed from the matrix triples. awk
# because the fields of one entry span several lines: a triple is only complete
# at the ext line, which every entry carries.
published="$(awk '
    /^ *- target:/ { arch=""; plat=""; ext="" }
    /^ *arch:/  { arch=$2 }
    /^ *plat:/  { plat=$2 }
    /^ *ext:/   { ext=$2; gsub(/"/, "", ext)
                  if (arch && plat) print arch "-" plat ext }
' "$workflow" | sort -u)"
if [ -z "$published" ]; then
    echo "could not read the matrix triples from $workflow" >&2
    exit 1
fi

# uname pairs a real machine reports, and the triple each must resolve to. The
# left side is what `uname -m` and `uname -s` print. Nothing here asserts an
# asset name by hand; the assertion is that it appears in the matrix above.
cases="
x86_64|Linux|x86_64-linux
amd64|Linux|x86_64-linux
aarch64|Linux|aarch64-linux
arm64|Linux|aarch64-linux
x86_64|Darwin|x86_64-macos
arm64|Darwin|aarch64-macos
aarch64|Darwin|aarch64-macos
x86_64|MINGW64_NT-10.0-22631|x86_64-windows.exe
x86_64|MSYS_NT-10.0-19045|x86_64-windows.exe
x86_64|CYGWIN_NT-10.0|x86_64-windows.exe
arm64|MINGW64_NT-10.0|aarch64-windows.exe
aarch64|Windows_NT|aarch64-windows.exe
"

fake_uname_dir() {
    d="$(mktemp -d)"
    cat >"$d/uname" <<EOF
#!/bin/sh
case "\$1" in
    -m) echo "$1" ;;
    -s) echo "$2" ;;
    *)  echo "$2" ;;
esac
EOF
    chmod +x "$d/uname"
    printf '%s\n' "$d"
}

fail=0
checked=0

while IFS='|' read -r arch os triple; do
    [ -n "${arch:-}" ] || continue
    checked=$((checked + 1))

    d="$(fake_uname_dir "$arch" "$os")"
    got_launch="$(PATH="$d:$PATH" sh "$here/bin/distil-mcp-launch" --artifact 2>/dev/null)"
    got_install="$(PATH="$d:$PATH" bash "$root/install.sh" --print-artifact 2>/dev/null)"
    rm -rf "$d"

    # The launcher speaks for distil-mcp only; install.sh prints one line per
    # binary, and every one of them has to land on the same triple.
    check() {
        who="$1" got="$2" want="$3"
        if [ "$got" != "$want" ]; then
            printf 'FAIL %-11s %-22s %-6s -> %s (want %s)\n' \
                "$who" "$os" "$arch" "${got:-<none>}" "$want"
            fail=$((fail + 1))
            return
        fi
        # The name must be one the release really publishes, not merely the one
        # the two scripts agree on. Both can agree and both be wrong.
        if ! printf '%s\n' "$published" | grep -qx "$triple"; then
            printf 'FAIL %-11s %-22s %-6s -> %s is not published by the matrix\n' \
                "$who" "$os" "$arch" "$got"
            fail=$((fail + 1))
        fi
    }

    check "launcher" "$got_launch" "${binaries[0]}-$triple"

    i=0
    while IFS= read -r line; do
        [ -n "$line" ] || continue
        check "install.sh" "$line" "${binaries[$i]}-$triple"
        i=$((i + 1))
    done <<< "$got_install"
    if [ "$i" -ne "${#binaries[@]}" ]; then
        printf 'FAIL install.sh  %-22s %-6s printed %d name(s), expected %d\n' \
            "$os" "$arch" "$i" "${#binaries[@]}"
        fail=$((fail + 1))
    fi
done <<EOF
$(printf '%s\n' "$cases")
EOF

# Every published triple must be reachable from some real platform. An asset
# nobody can resolve is build time spent on a download that never happens.
while IFS= read -r triple; do
    [ -n "$triple" ] || continue
    if ! printf '%s\n' "$cases" | grep -q "|$triple\$"; then
        printf 'FAIL unreachable  %s is published but no uname pair resolves to it\n' "$triple"
        fail=$((fail + 1))
    fi
done <<EOF
$published
EOF

# The npm wrapper is another place this mapping is written, and it speaks a
# different vocabulary for the same six assets: Node says darwin/win32/x64 where
# uname says Darwin/MINGW64_NT/x86_64. Two scripts agreeing proved nothing when
# both were wrong about a Mac, so the wrapper is held to the same matrix.
node_checked=0
if command -v node >/dev/null 2>&1 && [ -f "$root/npm/bin/selftest.js" ]; then
    if ! node "$root/npm/bin/selftest.js" >/dev/null 2>&1; then
        printf 'FAIL npm wrapper  bin/selftest.js failed its own assertions\n'
        fail=$((fail + 1))
    fi
    node_table="$(node "$root/npm/bin/selftest.js" 2>/dev/null)"
    while IFS="$(printf '\t')" read -r platform arch asset; do
        [ -n "${asset:-}" ] || continue
        node_checked=$((node_checked + 1))
        checked=$((checked + 1))
        triple="${asset#distil-mcp-}"
        if ! printf '%s\n' "$published" | grep -qx "$triple"; then
            printf 'FAIL npm wrapper  %s/%s -> %s (%s) is not in the release matrix\n' \
                "$platform" "$arch" "$asset" "$triple"
            fail=$((fail + 1))
        fi
    done <<EOF
$node_table
EOF
else
    printf '>>> SKIPPED the npm wrapper check (no node, or no npm/bin/selftest.js) <<<\n' >&2
fi

if [ "$fail" -gt 0 ]; then
    printf '\n%d platform failure(s) across %d case(s)\n' "$fail" "$checked" >&2
    exit 1
fi
printf 'platform mapping: %d case(s), both scripts match the release matrix\n' "$checked"
