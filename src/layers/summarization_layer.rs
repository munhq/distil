use crate::counter::TokenCounter;
use crate::pipeline::{Ctx, Layer, LayerResult};
use crate::summarizer::Summarizer;
use crate::types::{Message, Role};

/// Semantically compresses old conversation turns using a caller-provided LLM.
///
/// Unlike other layers which are purely structural, `SummarizationLayer` calls
/// an external summarization function to produce a dense summary of old turns.
/// JetBrains measured +7-11% savings on top of structural masking.
///
/// The caller provides a [`Summarizer`] implementation — distil decides *when*
/// to invoke it and injects the result.
///
/// # Pipeline ordering
///
/// Place after `MaskingLayer` (so already-masked results aren't re-summarized)
/// and before `BudgetLayer` (so the budget trimmer sees the smaller context).
///
/// ```rust,ignore
/// Pipeline::builder()
///     .layer(RegistryLayer::new(tools, &counter))
///     .layer(MaskingLayer::new())
///     .layer(SummarizationLayer::new(my_summarizer))  // ← here
///     .layer(CompactionLayer::new())
///     .layer(BudgetLayer::new(32_000))
///     .build();
/// ```
pub struct SummarizationLayer<S: Summarizer> {
    summarizer: S,
    /// Only summarize turns older than this from current turn (default: 4).
    age_threshold: u32,
    /// Maximum tokens for the summary output (default: 200).
    max_summary_tokens: usize,
    /// Don't bother summarizing if old content is below this (default: 100).
    min_content_tokens: usize,
}

impl<S: Summarizer> SummarizationLayer<S> {
    pub fn new(summarizer: S) -> Self {
        Self {
            summarizer,
            age_threshold: 4,
            max_summary_tokens: 200,
            min_content_tokens: 100,
        }
    }

    /// Set the age threshold: only turns older than this are summarized (default: 4).
    pub fn age_threshold(mut self, turns: u32) -> Self {
        self.age_threshold = turns;
        self
    }

    /// Set the maximum tokens for the summary output (default: 200).
    pub fn max_summary_tokens(mut self, tokens: usize) -> Self {
        self.max_summary_tokens = tokens;
        self
    }

    /// Set minimum tokens in old content before summarization kicks in (default: 100).
    pub fn min_content_tokens(mut self, tokens: usize) -> Self {
        self.min_content_tokens = tokens;
        self
    }
}

impl<S: Summarizer + 'static> Layer for SummarizationLayer<S> {
    fn name(&self) -> &str {
        "summarization"
    }

    fn apply(&self, ctx: &mut Ctx, counter: &dyn TokenCounter) -> LayerResult {
        let tokens_before = ctx.total_tokens(counter);
        let cutoff = ctx.turn.saturating_sub(self.age_threshold);

        // If cutoff is 0, nothing is old enough to summarize
        if cutoff == 0 {
            return LayerResult {
                layer: self.name().into(),
                tokens_before,
                tokens_after: tokens_before,
                detail: "no turns old enough to summarize".into(),
            };
        }

        // Track turns and collect indices of old non-system messages
        let mut turn: u32 = 0;
        let mut seen_first_user = false;
        let mut old_indices: Vec<usize> = Vec::new();
        let mut old_content = String::new();

        for (i, msg) in ctx.messages.iter().enumerate() {
            if msg.role == Role::User {
                if seen_first_user {
                    turn += 1;
                }
                seen_first_user = true;
            }

            if msg.role == Role::System {
                continue;
            }

            if turn < cutoff {
                old_indices.push(i);
                let role_str = match msg.role {
                    Role::User => "user",
                    Role::Assistant => "assistant",
                    Role::Tool => "tool",
                    Role::System => unreachable!(),
                };
                old_content.push_str(&format!("[{}]: {}\n", role_str, msg.content));
            }
        }

        if old_indices.is_empty() {
            return LayerResult {
                layer: self.name().into(),
                tokens_before,
                tokens_after: tokens_before,
                detail: "no old messages to summarize".into(),
            };
        }

        let old_tokens = counter.count(&old_content);
        if old_tokens < self.min_content_tokens {
            return LayerResult {
                layer: self.name().into(),
                tokens_before,
                tokens_after: tokens_before,
                detail: format!(
                    "old content too small ({old_tokens} < {} tokens), skipped",
                    self.min_content_tokens
                ),
            };
        }

        // Call the summarizer
        let summary = match self.summarizer.summarize(&old_content, self.max_summary_tokens) {
            Ok(s) => s,
            Err(e) => {
                return LayerResult {
                    layer: self.name().into(),
                    tokens_before,
                    tokens_after: tokens_before,
                    detail: format!("summarization failed: {e}, context unchanged"),
                };
            }
        };

        // Remove old messages (in reverse order to preserve indices)
        for &idx in old_indices.iter().rev() {
            ctx.messages.remove(idx);
        }

        // Insert summary after the first system message (or at position 0)
        let insert_pos = ctx
            .messages
            .iter()
            .position(|m| m.role != Role::System)
            .unwrap_or(ctx.messages.len());

        ctx.messages.insert(
            insert_pos,
            Message::system(format!("## Conversation Summary\n{summary}")),
        );

        let tokens_after = ctx.total_tokens(counter);

        LayerResult {
            layer: self.name().into(),
            tokens_before,
            tokens_after,
            detail: format!(
                "summarized {} old messages ({old_tokens} → {} tokens)",
                old_indices.len(),
                counter.count(&summary)
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::counter::EstimateCounter;

    struct MockSummarizer {
        response: String,
        should_fail: bool,
    }

    impl MockSummarizer {
        fn ok(response: &str) -> Self {
            Self {
                response: response.into(),
                should_fail: false,
            }
        }

        fn failing() -> Self {
            Self {
                response: String::new(),
                should_fail: true,
            }
        }
    }

    impl Summarizer for MockSummarizer {
        fn summarize(&self, _content: &str, _max_tokens: usize) -> crate::error::Result<String> {
            if self.should_fail {
                Err(crate::Error::Summarization("mock failure".into()))
            } else {
                Ok(self.response.clone())
            }
        }
    }

    #[test]
    fn summarizes_old_turns() {
        let counter = EstimateCounter;
        let layer = SummarizationLayer::new(MockSummarizer::ok(
            "User asked to build auth. JWT module was created and tests pass.",
        ))
        .age_threshold(2)
        .min_content_tokens(10);

        let mut ctx = Ctx::new(
            vec![
                Message::system("You are a coding assistant."),
                // Turn 0 (old)
                Message::user("Build the auth module"),
                Message::assistant("I'll create src/auth.rs with JWT validation. ".repeat(20)),
                // Turn 1 (old)
                Message::user("Run the tests"),
                Message::assistant("All 12 tests pass. ".repeat(10)),
                // Turn 2 (recent)
                Message::user("Now add rate limiting"),
                Message::assistant("Adding rate limiter..."),
                // Turn 3 (current)
                Message::user("What's next?"),
            ],
            vec![],
            3,
        );

        let result = layer.apply(&mut ctx, &counter);

        assert!(result.tokens_saved() > 0, "should save tokens: {result}");
        assert!(
            result.detail.contains("summarized"),
            "detail should mention summarization: {}",
            result.detail
        );

        // Summary message should exist
        let has_summary = ctx
            .messages
            .iter()
            .any(|m| m.content.contains("## Conversation Summary"));
        assert!(has_summary, "should have injected summary message");

        // Recent messages should be untouched
        let has_rate_limiting = ctx.messages.iter().any(|m| m.content.contains("rate limiting"));
        assert!(has_rate_limiting, "recent messages should be preserved");

        // Old messages should be gone
        let has_auth_module = ctx
            .messages
            .iter()
            .any(|m| m.content.contains("Build the auth module"));
        assert!(!has_auth_module, "old messages should be removed");
    }

    #[test]
    fn skips_when_content_too_small() {
        let counter = EstimateCounter;
        let layer = SummarizationLayer::new(MockSummarizer::ok("summary"))
            .age_threshold(1)
            .min_content_tokens(1000); // very high threshold

        let mut ctx = Ctx::new(
            vec![
                Message::system("System"),
                Message::user("Hi"),
                Message::assistant("Hello"),
                Message::user("More"),
                Message::assistant("Sure"),
                Message::user("Bye"),
            ],
            vec![],
            2,
        );

        let original_len = ctx.messages.len();
        let result = layer.apply(&mut ctx, &counter);

        assert_eq!(ctx.messages.len(), original_len, "messages should be unchanged");
        assert!(result.detail.contains("too small"), "detail: {}", result.detail);
    }

    #[test]
    fn handles_summarizer_failure_gracefully() {
        let counter = EstimateCounter;
        let layer = SummarizationLayer::new(MockSummarizer::failing())
            .age_threshold(1)
            .min_content_tokens(5);

        let mut ctx = Ctx::new(
            vec![
                Message::system("System"),
                Message::user("Build something big"),
                Message::assistant("Done. ".repeat(50)),
                Message::user("Continue"),
                Message::assistant("More work. ".repeat(20)),
                Message::user("Next?"),
            ],
            vec![],
            2,
        );

        let original_len = ctx.messages.len();
        let result = layer.apply(&mut ctx, &counter);

        assert_eq!(ctx.messages.len(), original_len, "messages unchanged on failure");
        assert!(
            result.detail.contains("failed"),
            "detail should mention failure: {}",
            result.detail
        );
        assert_eq!(result.tokens_saved(), 0);
    }

    #[test]
    fn preserves_system_messages() {
        let counter = EstimateCounter;
        let layer = SummarizationLayer::new(MockSummarizer::ok("summary"))
            .age_threshold(1)
            .min_content_tokens(5);

        let mut ctx = Ctx::new(
            vec![
                Message::system("You are a coding assistant. Follow best practices."),
                Message::user("Build auth"),
                Message::assistant("Done building auth module with JWT. ".repeat(10)),
                Message::user("Run tests"),
                Message::assistant("All tests pass. ".repeat(5)),
                Message::user("What's next?"),
            ],
            vec![],
            2,
        );

        let result = layer.apply(&mut ctx, &counter);

        // Original system message should still be there
        assert_eq!(ctx.messages[0].role, Role::System);
        assert!(
            ctx.messages[0].content.contains("coding assistant"),
            "original system message preserved: {}",
            ctx.messages[0].content
        );

        // Summary should also be a system message
        let summary_msgs: Vec<_> = ctx
            .messages
            .iter()
            .filter(|m| m.content.contains("Conversation Summary"))
            .collect();
        assert_eq!(summary_msgs.len(), 1, "should have exactly one summary");
        assert_eq!(summary_msgs[0].role, Role::System);

        let _ = result;
    }
}
