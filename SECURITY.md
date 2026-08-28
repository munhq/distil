# Security policy

## Supported versions

The latest released version is supported. Fixes go into a new release rather
than into a patch of an older one.

## Reporting a vulnerability

Report privately through GitHub Security Advisories:

**https://github.com/munhq/distil/security/advisories/new**

Do not open a public issue for a vulnerability. Expect an acknowledgement within
seven days, and an assessment with a fix or a rejection within thirty.

Please include the version, the platform, the smallest input that reproduces the
problem, and what an attacker gains. Do not include real credentials or real
transcript content in the report — describe the shape of the input instead.

## What distil touches

Read this before deciding whether a finding is in scope.

`distil-mcp` and `distil-proxy` operate on the conversation a client sends them.
They read no files, and `distil-mcp` makes no network calls at all. Everything
about a request is supplied by the caller, so a hostile conversation is the
realistic threat: pathological regular-expression input to `MaskingLayer`, deeply
nested JSON to the truncator, or an oversized message to the counters.

`distil-bench` and `distil-probe` are the exception. They read agent transcripts
from disk — by default `~/.claude/projects` — and those transcripts contain
whatever the agent handled, which may include secrets. They write only where
`--json`, `--export-sessions` or `--export-payloads` tells them to. Treat any
exported corpus as sensitive, and do not commit one.

`SummarizationLayer` and `distil-probe` call an LLM only through a `Summarizer`
or `Completer` the caller supplies, with an endpoint and a key the caller
provides. distil never embeds a key and never picks an endpoint of its own.

## Install-path integrity

`install.sh`, the npm wrapper and the Dockerfile all download a release asset and
verify it against the `checksums.txt` published beside it in the same release,
and they fail rather than run an unverified binary. A report that any of those
three paths can be made to execute an unverified download is in scope.
