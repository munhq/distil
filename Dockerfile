# distil as a container, for hosts that install MCP servers as images.
#
# The release binary is a musl static build, so this image exists to satisfy those
# hosts rather than because distil needs a runtime. It fetches the release asset
# for the target architecture and verifies it against the checksums published
# beside it, the same contract the npm wrapper and install.sh follow.
#
#   docker run -i --rm munhq/distil
#
# No volume and no network: distil optimises the conversation a client sends over
# stdio. It reads no files and calls nothing out.
ARG VERSION=0.3.1

FROM alpine:3.21 AS fetch
ARG VERSION
ARG TARGETARCH
RUN apk add --no-cache ca-certificates curl
WORKDIR /out
# TARGETARCH is Docker's vocabulary (amd64/arm64); the release names assets by
# arch and platform. The mapping is spelled out rather than assembled — an
# assembled one is how a platform ends up asking for a name that does not exist.
RUN set -eu; \
    case "$TARGETARCH" in \
      amd64) ASSET="distil-mcp-x86_64-linux" ;; \
      arm64) ASSET="distil-mcp-aarch64-linux" ;; \
      *) echo "no distil release build for TARGETARCH=$TARGETARCH" >&2; exit 1 ;; \
    esac; \
    BASE="https://github.com/munhq/distil/releases/download/v${VERSION}"; \
    curl -fsSL -o distil-mcp "$BASE/$ASSET"; \
    curl -fsSL -o checksums.txt "$BASE/checksums.txt"; \
    WANT="$(awk -v a="$ASSET" '$2==a{print $1}' checksums.txt)"; \
    [ -n "$WANT" ] || { echo "checksums.txt for v${VERSION} does not list $ASSET" >&2; exit 1; }; \
    printf '%s  distil-mcp\n' "$WANT" | sha256sum -c -; \
    chmod 0755 distil-mcp; \
    rm checksums.txt

# The binary and nothing else: no shell to escalate into, no package manager to
# drift.
FROM scratch
ARG VERSION
LABEL org.opencontainers.image.title="distil" \
      org.opencontainers.image.description="Context optimization middleware for LLM agents: dynamic tool registry, result masking, token budgeting and smart compaction." \
      org.opencontainers.image.source="https://github.com/munhq/distil" \
      org.opencontainers.image.licenses="MIT OR Apache-2.0" \
      org.opencontainers.image.version="${VERSION}"
COPY --from=fetch /out/distil-mcp /distil-mcp
# stdio: the client speaks JSON-RPC over this container's stdin and stdout, so
# `docker run -i` is required and nothing may be printed to stdout but protocol.
ENTRYPOINT ["/distil-mcp"]
