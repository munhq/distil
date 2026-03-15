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
