/// Default system prompt for conversation summarization.
///
/// Follows Anthropic's context engineering guidance: preserve architectural decisions,
/// unresolved bugs, implementation details, and file paths. Discard redundant tool
/// outputs, verbose build logs, and repetitive confirmation messages.
pub const DEFAULT_SUMMARIZER_SYSTEM_PROMPT: &str = "\
You are a conversation summarizer for an LLM coding agent. \
Produce a dense, factual summary that another LLM can use to continue the work. \
No preamble — output only the summary.\n\n\
PRESERVE (high signal):\n\
- Architectural decisions and their rationale\n\
- File paths created, modified, or deleted\n\
- Commands run and their meaningful outcomes (pass/fail, error messages)\n\
- Unresolved issues, bugs, or open questions\n\
- Configuration values, environment variables, API endpoints\n\
- Key data: schema structures, type definitions, function signatures\n\n\
DISCARD (low signal):\n\
- Verbose build/compile logs (just note success/failure + warning count)\n\
- Full tool output JSON (summarize the result, not the raw data)\n\
- Repetitive confirmation messages\n\
- Intermediate reasoning that led to a final decision (keep the decision)\n\
- File contents that were read but not modified";

/// Trait for semantic summarization of conversation content.
///
/// Distil is sync and LLM-agnostic — it never calls an LLM directly.
/// The caller provides a `Summarizer` implementation that wraps their
/// preferred LLM client.
///
/// If the implementation needs async (e.g., HTTP API call), use
/// `tokio::task::block_in_place` + `Handle::current().block_on()` inside
/// the `summarize` method.
///
/// # Example
///
/// ```rust,ignore
/// struct MySummarizer { client: MyLlmClient }
///
/// impl distil::Summarizer for MySummarizer {
///     fn summarize(&self, content: &str, max_tokens: usize) -> distil::error::Result<String> {
///         let prompt = format!(
///             "Summarize in under {max_tokens} tokens:\n{content}"
///         );
///         self.client
///             .complete(&prompt)
///             .map_err(|e| distil::Error::Summarization(e.to_string()))
///     }
/// }
/// ```
pub trait Summarizer: Send + Sync {
    /// Summarize the given content into at most `max_tokens` tokens.
    ///
    /// The content is a concatenation of old conversation turns formatted as
    /// `[role]: content\n`. The implementation should return a concise summary
    /// preserving key decisions, outcomes, and context.
    fn summarize(&self, content: &str, max_tokens: usize) -> crate::error::Result<String>;
}

// ── HttpSummarizer — real LLM-backed summarizer ─────────────────────────────

/// Calls any OpenAI-compatible chat/completions endpoint to produce summaries.
///
/// Uses blocking HTTP inside `tokio::task::block_in_place` so it works from
/// sync `Layer::apply` even when called within an async runtime.
///
/// # Example
///
/// ```rust,ignore
/// let summarizer = HttpSummarizer::new(
///     "https://api.anthropic.com/v1/messages",
///     "claude-haiku-4-5-20251001",
///     std::env::var("ANTHROPIC_API_KEY").unwrap(),
/// );
/// let pipeline = Pipeline::builder()
///     .layer(SummarizationLayer::new(summarizer))
///     .build();
/// ```
#[cfg(feature = "proxy")]
pub struct HttpSummarizer {
    endpoint: String,
    model: String,
    api_key: String,
    client: reqwest::blocking::Client,
}

#[cfg(feature = "proxy")]
impl HttpSummarizer {
    pub fn new(endpoint: impl Into<String>, model: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into(),
            model: model.into(),
            api_key: api_key.into(),
            client: reqwest::blocking::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .build()
                .expect("failed to build HTTP client"),
        }
    }

    /// Try to create from environment variables. Returns `None` if any required
    /// var is missing.
    ///
    /// - `DISTIL_SUMMARIZER_ENDPOINT`
    /// - `DISTIL_SUMMARIZER_MODEL`
    /// - `DISTIL_SUMMARIZER_API_KEY`
    pub fn from_env() -> Option<Self> {
        let endpoint = std::env::var("DISTIL_SUMMARIZER_ENDPOINT").ok()?;
        let model = std::env::var("DISTIL_SUMMARIZER_MODEL").ok()?;
        let api_key = std::env::var("DISTIL_SUMMARIZER_API_KEY").ok()?;
        Some(Self::new(endpoint, model, api_key))
    }
}

#[cfg(feature = "proxy")]
impl Summarizer for HttpSummarizer {
    fn summarize(&self, content: &str, max_tokens: usize) -> crate::error::Result<String> {
        let body = serde_json::json!({
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [
                {
                    "role": "system",
                    "content": DEFAULT_SUMMARIZER_SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": format!("Summarize this conversation in under {max_tokens} tokens:\n\n{content}")
                }
            ]
        });

        let do_request = || -> Result<String, String> {
            let resp = self
                .client
                .post(&self.endpoint)
                .header("Content-Type", "application/json")
                .header("Authorization", format!("Bearer {}", self.api_key))
                .header("x-api-key", &self.api_key) // Anthropic uses this header
                .json(&body)
                .send()
                .map_err(|e| format!("HTTP request failed: {e}"))?;

            let status = resp.status();
            let text = resp.text().map_err(|e| format!("failed to read response: {e}"))?;

            if !status.is_success() {
                return Err(format!("HTTP {status}: {text}"));
            }

            let json: serde_json::Value =
                serde_json::from_str(&text).map_err(|e| format!("invalid JSON: {e}"))?;

            // OpenAI format: choices[0].message.content
            if let Some(content) = json["choices"][0]["message"]["content"].as_str() {
                return Ok(content.to_string());
            }
            // Anthropic format: content[0].text
            if let Some(content) = json["content"][0]["text"].as_str() {
                return Ok(content.to_string());
            }

            Err(format!("unexpected response format: {text}"))
        };

        // If we're inside a tokio runtime, use block_in_place to avoid blocking
        // the async executor. Otherwise just run directly.
        let result = if tokio::runtime::Handle::try_current().is_ok() {
            tokio::task::block_in_place(do_request)
        } else {
            do_request()
        };

        result.map_err(crate::Error::Summarization)
    }
}

// ── OllamaSummarizer — local Ollama instance ────────────────────────────────

/// Calls a local Ollama instance for summarization.
/// Much faster than remote APIs (typically <1 second).
///
/// # Example
///
/// ```rust,ignore
/// let summarizer = OllamaSummarizer::new("qwen2.5:3b");
/// let pipeline = Pipeline::builder()
///     .layer(SummarizationLayer::new(summarizer))
///     .build();
/// ```
#[cfg(feature = "proxy")]
pub struct OllamaSummarizer {
    endpoint: String,
    model: String,
    client: reqwest::blocking::Client,
}

#[cfg(feature = "proxy")]
impl OllamaSummarizer {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            endpoint: "http://localhost:11434".into(),
            model: model.into(),
            client: reqwest::blocking::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .build()
                .expect("failed to build HTTP client"),
        }
    }

    pub fn endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = endpoint.into();
        self
    }
}

#[cfg(feature = "proxy")]
impl Summarizer for OllamaSummarizer {
    fn summarize(&self, content: &str, max_tokens: usize) -> crate::error::Result<String> {
        let body = serde_json::json!({
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": DEFAULT_SUMMARIZER_SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": format!("Summarize this conversation in under {max_tokens} tokens:\n\n{content}")
                }
            ],
            "stream": false,
            "options": {
                "num_predict": max_tokens as i64
            }
        });

        let url = format!("{}/api/chat", self.endpoint);

        let do_request = || -> Result<String, String> {
            let resp = self.client
                .post(&url)
                .json(&body)
                .send()
                .map_err(|e| format!("Ollama request failed: {e}"))?;

            let status = resp.status();
            let text = resp.text().map_err(|e| format!("failed to read response: {e}"))?;

            if !status.is_success() {
                return Err(format!("Ollama HTTP {status}: {text}"));
            }

            let json: serde_json::Value =
                serde_json::from_str(&text).map_err(|e| format!("invalid JSON: {e}"))?;

            // Ollama chat format: message.content
            json["message"]["content"]
                .as_str()
                .map(|s| s.to_string())
                .ok_or_else(|| format!("unexpected Ollama response: {text}"))
        };

        // If we're inside a tokio runtime, use block_in_place to avoid blocking
        // the async executor. Otherwise just run directly.
        let result = if tokio::runtime::Handle::try_current().is_ok() {
            tokio::task::block_in_place(do_request)
        } else {
            do_request()
        };

        result.map_err(crate::Error::Summarization)
    }
}
