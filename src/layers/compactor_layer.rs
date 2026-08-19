use std::time::Duration;

use crate::counter::TokenCounter;
use crate::pipeline::{Ctx, Layer, LayerResult};
use crate::types::{Message, Role};

/// Intelligent context compaction without LLM summarization.
///
/// This layer reduces context size through structural analysis:
/// - Collapses consecutive messages of the same role
/// - Removes empty or near-empty messages
/// - Deduplicates repeated content (e.g., same error appearing multiple times)
/// - Strips verbose formatting (excessive whitespace, decorative separators)
///
/// For LLM-powered summarization, the caller should summarize externally
/// and replace old messages before running the pipeline. Distil stays
/// LLM-agnostic — it never calls an LLM itself.
pub struct CompactionLayer {
    /// Merge consecutive messages from the same role.
    merge_consecutive: bool,
    /// Remove messages with fewer tokens than this.
    min_message_tokens: usize,
    /// Collapse runs of whitespace in message content.
    strip_whitespace: bool,
    /// Deduplicate identical content within N messages.
    dedup_window: usize,
}

impl CompactionLayer {
    pub fn new() -> Self {
        Self {
            merge_consecutive: true,
            min_message_tokens: 2,
            strip_whitespace: true,
            dedup_window: 10,
        }
    }

    pub fn merge_consecutive(mut self, enabled: bool) -> Self {
        self.merge_consecutive = enabled;
        self
    }

    pub fn min_message_tokens(mut self, min: usize) -> Self {
        self.min_message_tokens = min;
        self
    }

    pub fn strip_whitespace(mut self, enabled: bool) -> Self {
        self.strip_whitespace = enabled;
        self
    }

    pub fn dedup_window(mut self, window: usize) -> Self {
        self.dedup_window = window;
        self
    }
}

impl Default for CompactionLayer {
    fn default() -> Self {
        Self::new()
    }
}

impl Layer for CompactionLayer {
    fn name(&self) -> &str {
        "compactor"
    }

    fn phase(&self) -> Option<crate::pipeline::Phase> {
        Some(crate::pipeline::Phase::Compress)
    }

    fn apply(&self, ctx: &mut Ctx, counter: &dyn TokenCounter) -> LayerResult {
        let tokens_before = ctx.total_tokens(counter);
        let msg_count_before = ctx.messages.len();

        // Step 1: Strip excessive whitespace
        if self.strip_whitespace {
            for msg in &mut ctx.messages {
                msg.content = compact_whitespace(&msg.content);
            }
        }

        // Step 2: Remove near-empty messages (but never system messages)
        ctx.messages.retain(|msg| {
            msg.role == Role::System || counter.count(&msg.content) >= self.min_message_tokens
        });

        // Step 3: Deduplicate identical content within a sliding window
        if self.dedup_window > 0 {
            ctx.messages = dedup_messages(&ctx.messages, self.dedup_window);
        }

        // Step 4: Merge consecutive same-role messages
        if self.merge_consecutive {
            ctx.messages = merge_consecutive_messages(&ctx.messages);
        }

        let tokens_after = ctx.total_tokens(counter);
        let removed = msg_count_before.saturating_sub(ctx.messages.len());

        LayerResult {
            layer: self.name().into(),
            tokens_before,
            tokens_after,
            duration: Duration::ZERO,
            detail: format!(
                "compacted {msg_count_before} → {} messages ({removed} removed)",
                ctx.messages.len()
            ),
        }
    }
}

/// Collapse runs of 3+ newlines into 2, and runs of 3+ spaces into 1.
fn compact_whitespace(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut newline_count = 0;
    let mut space_count = 0;

    for ch in s.chars() {
        match ch {
            '\n' => {
                newline_count += 1;
                space_count = 0;
                if newline_count <= 2 {
                    result.push(ch);
                }
            }
            ' ' | '\t' => {
                newline_count = 0;
                space_count += 1;
                if space_count <= 2 {
                    result.push(if ch == '\t' { ' ' } else { ch });
                }
            }
            _ => {
                newline_count = 0;
                space_count = 0;
                result.push(ch);
            }
        }
    }

    result
}

/// Remove duplicate messages within a sliding window.
/// Keeps the first occurrence, removes later duplicates.
fn dedup_messages(messages: &[Message], window: usize) -> Vec<Message> {
    let mut result = Vec::with_capacity(messages.len());

    for (i, msg) in messages.iter().enumerate() {
        // System messages are never deduped
        if msg.role == Role::System {
            result.push(msg.clone());
            continue;
        }

        // Check if this exact content appears in the preceding window
        let start = i.saturating_sub(window);
        let is_dup = messages[start..i]
            .iter()
            .any(|prev| prev.role == msg.role && prev.content == msg.content);

        if !is_dup {
            result.push(msg.clone());
        }
    }

    result
}

/// Merge consecutive messages of the same role (except System).
fn merge_consecutive_messages(messages: &[Message]) -> Vec<Message> {
    let mut result: Vec<Message> = Vec::with_capacity(messages.len());

    for msg in messages {
        if let Some(last) = result.last_mut() {
            if last.role == msg.role && last.role != Role::System {
                last.content.push('\n');
                last.content.push_str(&msg.content);
                continue;
            }
        }
        result.push(msg.clone());
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::counter::EstimateCounter;

    #[test]
    fn strips_excessive_whitespace() {
        let input = "line one\n\n\n\n\n\nline two\n\n\nline three";
        let result = compact_whitespace(input);
        assert_eq!(result, "line one\n\nline two\n\nline three");
    }

    #[test]
    fn merges_consecutive_same_role() {
        let messages = vec![
            Message::user("first"),
            Message::user("second"),
            Message::assistant("response"),
            Message::assistant("more response"),
        ];

        let merged = merge_consecutive_messages(&messages);
        assert_eq!(merged.len(), 2);
        assert!(merged[0].content.contains("first"));
        assert!(merged[0].content.contains("second"));
        assert!(merged[1].content.contains("response"));
        assert!(merged[1].content.contains("more response"));
    }

    #[test]
    fn does_not_merge_system_messages() {
        let messages = vec![
            Message::system("system 1"),
            Message::system("system 2"),
            Message::user("hello"),
        ];

        let merged = merge_consecutive_messages(&messages);
        assert_eq!(merged.len(), 3);
    }

    #[test]
    fn deduplicates_within_window() {
        let messages = vec![
            Message::user("hello"),
            Message::assistant("hi"),
            Message::user("hello"),   // duplicate
            Message::assistant("hi"), // duplicate
        ];

        let deduped = dedup_messages(&messages, 5);
        assert_eq!(deduped.len(), 2);
    }

    #[test]
    fn removes_near_empty_messages() {
        let counter = EstimateCounter;
        let layer = CompactionLayer::new()
            .min_message_tokens(3)
            .merge_consecutive(false)
            .dedup_window(0);

        let mut ctx = Ctx::new(
            vec![
                Message::system("You are helpful."),
                Message::user("ok"), // ~1 token, below threshold
                Message::assistant("I understand."),
                Message::user(""), // empty
                Message::assistant("Sure, let me help with that task."),
            ],
            vec![],
            2,
        );

        layer.apply(&mut ctx, &counter);

        // System preserved, empty/tiny user messages removed
        assert!(ctx.messages.iter().any(|m| m.role == Role::System));
        assert!(
            ctx.messages
                .iter()
                .all(|m| !m.content.is_empty() || m.role == Role::System)
        );
    }

    #[test]
    fn full_compaction_reduces_tokens() {
        let counter = EstimateCounter;
        let layer = CompactionLayer::new();

        let verbose_content = format!("result:\n\n\n\n\n{}", "data\n".repeat(5));
        let mut ctx = Ctx::new(
            vec![
                Message::system("Be helpful."),
                Message::user("do thing"),
                Message::assistant(&verbose_content),
                Message::user("do thing"),            // duplicate
                Message::assistant(&verbose_content), // duplicate
                Message::user("now what?"),
            ],
            vec![],
            3,
        );

        let result = layer.apply(&mut ctx, &counter);
        assert!(result.tokens_saved() > 0, "should save tokens: {result}");
    }
}
