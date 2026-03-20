//! HTTP utilities for the distil proxy server.
//!
//! Pure functions extracted from `proxy.rs` so they can be unit tested
//! without starting an HTTP server.

/// RFC 9457-compliant error classification.
///
/// Maps HTTP status codes + error messages to structured error metadata
/// that agents can use for deterministic retry/abort decisions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ErrorClassification {
    /// Error category URI (e.g. `urn:distil:error:upstream-timeout`).
    pub error_type: &'static str,
    /// Short human-readable title.
    pub title: &'static str,
    /// Whether the client should retry.
    pub retryable: bool,
    /// Seconds to wait before retrying (only meaningful if `retryable` is true).
    pub retry_after: Option<u32>,
}

/// Classify an HTTP error by status code and message content.
///
/// Returns structured metadata following RFC 9457 principles. The message
/// is inspected for keywords (e.g. "timeout", "429", "rate limit") to
/// distinguish sub-categories within a status code.
pub fn classify_error(status: u16, message: &str) -> ErrorClassification {
    match status {
        401 => ErrorClassification {
            error_type: "urn:distil:error:unauthorized",
            title: "Unauthorized",
            retryable: false,
            retry_after: None,
        },
        400 => ErrorClassification {
            error_type: "urn:distil:error:bad-request",
            title: "Bad Request",
            retryable: false,
            retry_after: None,
        },
        502 => {
            let is_timeout = message.contains("timeout") || message.contains("timed out");
            let is_rate_limit = message.contains("429") || message.contains("rate limit");
            if is_rate_limit {
                ErrorClassification {
                    error_type: "urn:distil:error:upstream-rate-limit",
                    title: "Upstream Rate Limited",
                    retryable: true,
                    retry_after: Some(5),
                }
            } else if is_timeout {
                ErrorClassification {
                    error_type: "urn:distil:error:upstream-timeout",
                    title: "Upstream Timeout",
                    retryable: true,
                    retry_after: Some(2),
                }
            } else {
                ErrorClassification {
                    error_type: "urn:distil:error:upstream-error",
                    title: "Upstream Error",
                    retryable: true,
                    retry_after: Some(1),
                }
            }
        }
        500 => ErrorClassification {
            error_type: "urn:distil:error:internal",
            title: "Internal Server Error",
            retryable: true,
            retry_after: Some(1),
        },
        _ => ErrorClassification {
            error_type: "urn:distil:error:unknown",
            title: "Unknown Error",
            retryable: false,
            retry_after: None,
        },
    }
}

/// Build an RFC 9457-compliant JSON error body.
pub fn error_body(status: u16, message: &str) -> serde_json::Value {
    let c = classify_error(status, message);
    let mut body = serde_json::json!({
        "type": c.error_type,
        "title": c.title,
        "status": status,
        "detail": message,
        "retryable": c.retryable,
    });
    if let Some(secs) = c.retry_after {
        body["retry_after"] = serde_json::json!(secs);
    }
    body
}

/// Check a bearer token against an expected API key.
///
/// Returns `true` if:
/// - `expected_key` is `None` (auth not configured — allow all)
/// - The `authorization` header is `Bearer <expected_key>`
///
/// The `path` parameter is used to exempt health endpoints.
pub fn check_bearer_auth(
    authorization_header: Option<&str>,
    path: &str,
    expected_key: Option<&str>,
) -> bool {
    let expected = match expected_key {
        Some(k) => k,
        None => return true, // No auth configured
    };

    // Health endpoints are always public
    if path == "/health" || path == "/v1/health" {
        return true;
    }

    authorization_header
        .and_then(|v| v.strip_prefix("Bearer "))
        .is_some_and(|token| token == expected)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── classify_error tests ─────────────────────────────────────────────

    #[test]
    fn classify_bad_request() {
        let c = classify_error(400, "missing required field: messages");
        assert_eq!(c.error_type, "urn:distil:error:bad-request");
        assert_eq!(c.title, "Bad Request");
        assert!(!c.retryable);
        assert_eq!(c.retry_after, None);
    }

    #[test]
    fn classify_unauthorized() {
        let c = classify_error(401, "invalid token");
        assert_eq!(c.error_type, "urn:distil:error:unauthorized");
        assert!(!c.retryable);
    }

    #[test]
    fn classify_upstream_generic() {
        let c = classify_error(502, "upstream error: connection refused");
        assert_eq!(c.error_type, "urn:distil:error:upstream-error");
        assert!(c.retryable);
        assert_eq!(c.retry_after, Some(1));
    }

    #[test]
    fn classify_upstream_timeout() {
        let c = classify_error(502, "upstream error: request timed out after 30s");
        assert_eq!(c.error_type, "urn:distil:error:upstream-timeout");
        assert!(c.retryable);
        assert_eq!(c.retry_after, Some(2));
    }

    #[test]
    fn classify_upstream_timeout_keyword_variant() {
        let c = classify_error(502, "upstream error: timeout waiting for response");
        assert_eq!(c.error_type, "urn:distil:error:upstream-timeout");
    }

    #[test]
    fn classify_upstream_rate_limit_429() {
        let c = classify_error(502, "upstream error: HTTP 429 Too Many Requests");
        assert_eq!(c.error_type, "urn:distil:error:upstream-rate-limit");
        assert!(c.retryable);
        assert_eq!(c.retry_after, Some(5));
    }

    #[test]
    fn classify_upstream_rate_limit_keyword() {
        let c = classify_error(502, "upstream error: rate limit exceeded");
        assert_eq!(c.error_type, "urn:distil:error:upstream-rate-limit");
    }

    #[test]
    fn classify_internal_server_error() {
        let c = classify_error(500, "pipeline panic");
        assert_eq!(c.error_type, "urn:distil:error:internal");
        assert!(c.retryable);
        assert_eq!(c.retry_after, Some(1));
    }

    #[test]
    fn classify_unknown_status() {
        let c = classify_error(418, "I'm a teapot");
        assert_eq!(c.error_type, "urn:distil:error:unknown");
        assert!(!c.retryable);
    }

    // ── error_body tests ─────────────────────────────────────────────────

    #[test]
    fn error_body_has_rfc9457_fields() {
        let body = error_body(502, "upstream error: connection refused");
        assert_eq!(body["type"], "urn:distil:error:upstream-error");
        assert_eq!(body["title"], "Upstream Error");
        assert_eq!(body["status"], 502);
        assert_eq!(body["detail"], "upstream error: connection refused");
        assert_eq!(body["retryable"], true);
        assert_eq!(body["retry_after"], 1);
    }

    #[test]
    fn error_body_omits_retry_after_when_not_retryable() {
        let body = error_body(400, "bad request");
        assert_eq!(body["retryable"], false);
        assert!(body.get("retry_after").is_none() || body["retry_after"].is_null());
    }

    #[test]
    fn error_body_rate_limit_has_longer_retry() {
        let body = error_body(502, "429 rate limit");
        assert_eq!(body["retry_after"], 5);
    }

    // ── check_bearer_auth tests ──────────────────────────────────────────

    #[test]
    fn auth_disabled_allows_all() {
        assert!(check_bearer_auth(None, "/v1/optimize", None));
        assert!(check_bearer_auth(None, "/v1/chat/completions", None));
    }

    #[test]
    fn auth_health_always_public() {
        assert!(check_bearer_auth(None, "/health", Some("secret")));
        assert!(check_bearer_auth(None, "/v1/health", Some("secret")));
    }

    #[test]
    fn auth_valid_token() {
        assert!(check_bearer_auth(
            Some("Bearer my-secret-key"),
            "/v1/optimize",
            Some("my-secret-key"),
        ));
    }

    #[test]
    fn auth_invalid_token() {
        assert!(!check_bearer_auth(
            Some("Bearer wrong-key"),
            "/v1/optimize",
            Some("my-secret-key"),
        ));
    }

    #[test]
    fn auth_missing_header() {
        assert!(!check_bearer_auth(
            None,
            "/v1/optimize",
            Some("my-secret-key"),
        ));
    }

    #[test]
    fn auth_wrong_scheme() {
        assert!(!check_bearer_auth(
            Some("Basic dXNlcjpwYXNz"),
            "/v1/optimize",
            Some("my-secret-key"),
        ));
    }

    #[test]
    fn auth_empty_bearer() {
        assert!(!check_bearer_auth(
            Some("Bearer "),
            "/v1/optimize",
            Some("my-secret-key"),
        ));
    }

    #[test]
    fn auth_metrics_endpoint_requires_auth() {
        assert!(!check_bearer_auth(
            None,
            "/metrics",
            Some("my-secret-key"),
        ));
        assert!(check_bearer_auth(
            Some("Bearer my-secret-key"),
            "/metrics",
            Some("my-secret-key"),
        ));
    }
}
