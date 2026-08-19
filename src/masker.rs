use regex::Regex;
use serde_json::Value;

use crate::counter::TokenCounter;
use crate::types::{Message, Role};

// ── JSON truncation ──────────────────────────────────────────────────────────

/// Configuration for structural JSON truncation.
///
/// When a tool result contains valid JSON, instead of fully masking it we can
/// preserve the structure (keys, types) while truncating long values, large
/// arrays, and deep nesting. This retains semantic signal at a fraction of the
/// token cost.
#[derive(Debug, Clone)]
pub struct JsonTruncateConfig {
    /// Truncate string values longer than this (default: 80).
    pub max_string_len: usize,
    /// Keep at most this many array elements, summarize the rest (default: 3).
    pub max_array_items: usize,
    /// Replace objects nested deeper than this with `"[nested object]"` (default: 5).
    pub max_depth: usize,
    /// Minimum percentage savings required for JSON truncation to be used
    /// instead of falling back to full masking (default: 20.0).
    pub min_savings_pct: f64,
}

impl Default for JsonTruncateConfig {
    fn default() -> Self {
        Self {
            max_string_len: 80,
            max_array_items: 3,
            max_depth: 5,
            min_savings_pct: 20.0,
        }
    }
}

/// Recursively truncate a JSON value according to the config.
fn truncate_json_value(val: &Value, config: &JsonTruncateConfig, depth: usize) -> Value {
    match val {
        Value::String(s) => {
            if s.len() > config.max_string_len {
                // Truncate at char boundary
                let truncated: String = s.chars().take(config.max_string_len).collect();
                Value::String(format!("{}...[{} chars]", truncated, s.len()))
            } else {
                val.clone()
            }
        }
        Value::Array(arr) => {
            if arr.len() > config.max_array_items {
                let mut truncated: Vec<Value> = arr[..config.max_array_items]
                    .iter()
                    .map(|v| truncate_json_value(v, config, depth + 1))
                    .collect();
                let remaining = arr.len() - config.max_array_items;
                truncated.push(Value::String(format!("...and {remaining} more items")));
                Value::Array(truncated)
            } else {
                Value::Array(
                    arr.iter()
                        .map(|v| truncate_json_value(v, config, depth + 1))
                        .collect(),
                )
            }
        }
        Value::Object(map) => {
            if depth >= config.max_depth {
                Value::String("[nested object]".into())
            } else {
                Value::Object(
                    map.iter()
                        .map(|(k, v)| (k.clone(), truncate_json_value(v, config, depth + 1)))
                        .collect(),
                )
            }
        }
        // Numbers, bools, nulls pass through unchanged
        _ => val.clone(),
    }
}

// ── Result masker ────────────────────────────────────────────────────────────

/// Masks old tool results in conversation history to save tokens.
///
/// Tool outputs are often the single largest token consumer in agent conversations.
/// A single `shell` command can produce hundreds of lines of output that the LLM
/// saw once and doesn't need to re-read every turn.
///
/// The masker replaces old tool results with compact one-line summaries:
/// ```text
/// Before: <tool_result name="shell">... 847 lines of cargo build output ...</tool_result>
/// After:  [shell → 2,412 tokens, masked]
/// ```
///
/// If the tool result contains valid JSON, the masker can optionally preserve the
/// structure while truncating long values (see [`JsonTruncateConfig`]).
///
/// The JetBrains "Complexity Trap" research (NeurIPS 2025) showed that hiding
/// tool outputs saves ~50% of tokens with zero quality degradation on agent tasks.
pub struct ResultMasker {
    /// Only mask results older than this many turns from the current turn.
    retain_turns: u32,
    /// Override retention for `Role::Tool` messages (observations). If `None`, uses `retain_turns`.
    retain_turns_tool: Option<u32>,
    /// Override retention for `Role::Assistant` messages containing tool results. If `None`, uses `retain_turns`.
    retain_turns_assistant: Option<u32>,
    /// Mask any single result exceeding this token count, regardless of age.
    max_result_tokens: Option<usize>,
    /// JSON truncation config. When set, JSON tool results are structurally
    /// truncated before falling back to full masking.
    json_truncate: Option<JsonTruncateConfig>,
    /// Pattern to detect tool results in message content.
    patterns: Vec<ResultPattern>,
}

/// A pattern for detecting tool results in message content.
pub struct ResultPattern {
    /// Regex that matches a tool result block. Must have named groups:
    /// - `name`: the tool name
    /// - `output`: the tool output content
    regex: Regex,
    /// Format string for the masked replacement. Available placeholders:
    /// - `{name}`: tool name
    /// - `{tokens}`: token count of the original output
    replacement_fmt: String,
}

impl ResultMasker {
    /// Create a masker with default XML-tag pattern detection.
    pub fn new() -> Self {
        Self {
            retain_turns: 3,
            retain_turns_tool: None,
            retain_turns_assistant: None,
            max_result_tokens: None,
            json_truncate: Some(JsonTruncateConfig::default()),
            patterns: vec![ResultPattern::xml_tags(), ResultPattern::bracketed()],
        }
    }

    /// Set how many recent turns to keep unmasked.
    pub fn retain_turns(mut self, turns: u32) -> Self {
        self.retain_turns = turns;
        self
    }

    /// Set separate retention for `Role::Tool` messages (observations).
    /// Tool results are pure data and can be compressed more aggressively.
    pub fn retain_turns_tool(mut self, turns: u32) -> Self {
        self.retain_turns_tool = Some(turns);
        self
    }

    /// Set separate retention for `Role::Assistant` messages containing tool results.
    /// Assistant reasoning is needed for the LLM to follow its own logic.
    pub fn retain_turns_assistant(mut self, turns: u32) -> Self {
        self.retain_turns_assistant = Some(turns);
        self
    }

    /// Mask any single result exceeding this token count, even if recent.
    pub fn max_result_tokens(mut self, max: usize) -> Self {
        self.max_result_tokens = Some(max);
        self
    }

    /// Set JSON truncation config. Enabled by default.
    pub fn json_truncate(mut self, config: JsonTruncateConfig) -> Self {
        self.json_truncate = Some(config);
        self
    }

    /// Disable JSON truncation — always fall back to full masking.
    pub fn no_json_truncate(mut self) -> Self {
        self.json_truncate = None;
        self
    }

    /// Add a custom result detection pattern.
    pub fn add_pattern(mut self, pattern: ResultPattern) -> Self {
        self.patterns.push(pattern);
        self
    }

    /// Replace the default patterns entirely.
    pub fn with_patterns(mut self, patterns: Vec<ResultPattern>) -> Self {
        self.patterns = patterns;
        self
    }

    /// Mask tool results in the message history.
    ///
    /// Messages are numbered from 0 (oldest). `current_turn` is the turn index
    /// of the most recent exchange. Messages from turns older than
    /// `current_turn - retain_turns` will have their tool results masked.
    ///
    /// Returns the modified messages and the number of tokens saved.
    pub fn mask(
        &self,
        messages: &[Message],
        current_turn: u32,
        counter: &dyn TokenCounter,
    ) -> (Vec<Message>, usize) {
        let tool_retain = self.retain_turns_tool.unwrap_or(self.retain_turns);
        let assistant_retain = self.retain_turns_assistant.unwrap_or(self.retain_turns);
        let default_cutoff = current_turn.saturating_sub(self.retain_turns);
        let tool_cutoff = current_turn.saturating_sub(tool_retain);
        let assistant_cutoff = current_turn.saturating_sub(assistant_retain);

        let mut total_saved = 0;
        let mut turn: u32 = 0;
        let mut seen_first_user = false;

        let masked: Vec<Message> = messages
            .iter()
            .map(|msg| {
                // Track turns: each user message starts a new turn
                if msg.role == Role::User {
                    if seen_first_user {
                        turn += 1;
                    }
                    seen_first_user = true;
                }

                // Role::Tool messages — mask the entire content directly (no regex needed)
                if msg.role == Role::Tool {
                    let should_mask_tool = turn < tool_cutoff
                        || self
                            .max_result_tokens
                            .is_some_and(|max| counter.count(&msg.content) > max);

                    if should_mask_tool {
                        let original_tokens = counter.count(&msg.content);

                        // Try JSON truncation first
                        if let Some(ref jt_config) = self.json_truncate {
                            if let Ok(parsed) = serde_json::from_str::<Value>(&msg.content) {
                                let truncated_val = truncate_json_value(&parsed, jt_config, 0);
                                if let Ok(truncated_str) = serde_json::to_string(&truncated_val) {
                                    let truncated_tokens = counter.count(&truncated_str);
                                    let savings_pct = if original_tokens > 0
                                        && truncated_tokens < original_tokens
                                    {
                                        ((original_tokens - truncated_tokens) as f64
                                            / original_tokens as f64)
                                            * 100.0
                                    } else {
                                        0.0
                                    };
                                    if savings_pct >= jt_config.min_savings_pct {
                                        total_saved +=
                                            original_tokens.saturating_sub(truncated_tokens);
                                        return Message {
                                            role: msg.role,
                                            content: truncated_str,
                                        };
                                    }
                                }
                            }
                        }

                        // Full masking fallback
                        let replacement =
                            format!("[tool result → {original_tokens} tokens, masked]");
                        let replacement_tokens = counter.count(&replacement);
                        total_saved += original_tokens.saturating_sub(replacement_tokens);
                        return Message {
                            role: msg.role,
                            content: replacement,
                        };
                    }

                    return msg.clone();
                }

                // Determine age cutoff based on role
                let should_mask_by_age = match msg.role {
                    Role::Assistant => turn < assistant_cutoff,
                    Role::System => false,
                    _ => turn < default_cutoff,
                };

                // Apply regex-based masking for embedded tool results
                let mut content = msg.content.clone();
                let mut msg_saved = 0;

                for pattern in &self.patterns {
                    let result = pattern.mask_in(
                        &content,
                        should_mask_by_age,
                        self.max_result_tokens,
                        self.json_truncate.as_ref(),
                        counter,
                    );
                    msg_saved += result.1;
                    content = result.0;
                }

                total_saved += msg_saved;

                Message {
                    role: msg.role,
                    content,
                }
            })
            .collect();

        (masked, total_saved)
    }
}

impl Default for ResultMasker {
    fn default() -> Self {
        Self::new()
    }
}

impl ResultPattern {
    /// Pattern for XML-style tool results: `<tool_result name="X">output</tool_result>`
    pub fn xml_tags() -> Self {
        Self {
            regex: Regex::new(
                r#"(?s)<tool_result\s+name="(?P<name>[^"]+)">\s*(?P<output>.*?)\s*</tool_result>"#,
            )
            .expect("xml_tags regex is valid"),
            replacement_fmt: "[{name} → {tokens} tokens, masked]".into(),
        }
    }

    /// Pattern for bracket-style results: `[Tool: name]\noutput\n[/Tool]`
    pub fn bracketed() -> Self {
        Self {
            regex: Regex::new(r#"(?s)\[Tool:\s*(?P<name>[^\]]+)\]\s*(?P<output>.*?)\s*\[/Tool\]"#)
                .expect("bracketed regex is valid"),
            replacement_fmt: "[{name} → {tokens} tokens, masked]".into(),
        }
    }

    /// Create a custom pattern.
    pub fn custom(regex: &str, replacement_fmt: &str) -> Result<Self, regex::Error> {
        Ok(Self {
            regex: Regex::new(regex)?,
            replacement_fmt: replacement_fmt.into(),
        })
    }

    /// Apply masking to content. Returns (new_content, tokens_saved).
    fn mask_in(
        &self,
        content: &str,
        mask_by_age: bool,
        max_tokens: Option<usize>,
        json_truncate: Option<&JsonTruncateConfig>,
        counter: &dyn TokenCounter,
    ) -> (String, usize) {
        let mut result = content.to_string();
        let mut total_saved = 0;

        // Re-match after each replacement, because every substitution shifts the
        // offsets of everything after it.
        while let Some(caps) = self.regex.captures(&result) {
            let full_match = caps.get(0).unwrap();
            let name = caps.name("name").map(|m| m.as_str()).unwrap_or("unknown");
            let output = caps.name("output").map(|m| m.as_str()).unwrap_or("");

            let output_tokens = counter.count(output);
            let should_mask = mask_by_age || max_tokens.is_some_and(|max| output_tokens > max);

            if !should_mask {
                break;
            }

            // Try JSON truncation first
            if let Some(jt_config) = json_truncate {
                let output_trimmed = output.trim();
                if let Ok(parsed) = serde_json::from_str::<Value>(output_trimmed) {
                    let truncated_val = truncate_json_value(&parsed, jt_config, 0);
                    if let Ok(truncated_str) = serde_json::to_string(&truncated_val) {
                        let truncated_tokens = counter.count(&truncated_str);
                        let savings_pct = if output_tokens > 0 && truncated_tokens < output_tokens {
                            ((output_tokens - truncated_tokens) as f64 / output_tokens as f64)
                                * 100.0
                        } else {
                            0.0
                        };
                        if savings_pct >= jt_config.min_savings_pct {
                            let saved = output_tokens.saturating_sub(truncated_tokens);
                            total_saved += saved;
                            // Replace the entire match with a non-matchable format
                            // so the regex won't re-capture it on the next loop iteration
                            let replacement = format!("[{name} → json truncated]\n{truncated_str}");
                            result = format!(
                                "{}{}{}",
                                &result[..full_match.start()],
                                replacement,
                                &result[full_match.end()..]
                            );
                            continue;
                        }
                    }
                }
            }

            // Full masking fallback
            let replacement = self
                .replacement_fmt
                .replace("{name}", name)
                .replace("{tokens}", &output_tokens.to_string());

            let replacement_tokens = counter.count(&replacement);
            let saved = output_tokens.saturating_sub(replacement_tokens);
            total_saved += saved;

            result = format!(
                "{}{}{}",
                &result[..full_match.start()],
                replacement,
                &result[full_match.end()..]
            );
        }

        (result, total_saved)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::counter::EstimateCounter;

    fn make_tool_result(name: &str, output: &str) -> String {
        format!("<tool_result name=\"{name}\">\n{output}\n</tool_result>")
    }

    #[test]
    fn masks_old_xml_tool_results() {
        let counter = EstimateCounter;
        let masker = ResultMasker::new().retain_turns(1).no_json_truncate();

        let long_output = "line\n".repeat(200);
        let messages = vec![
            Message::system("You are a helpful assistant."),
            Message::user("Run cargo build"),
            Message::assistant(make_tool_result("shell", &long_output)),
            Message::user("Now run cargo test"),
            Message::assistant(make_tool_result("shell", "test output: all passed")),
            Message::user("What happened?"),
        ];

        let (masked, saved) = masker.mask(&messages, 2, &counter);

        assert!(
            masked[2].content.contains("masked"),
            "old result should be masked: {}",
            masked[2].content
        );
        assert!(
            masked[4].content.contains("all passed"),
            "recent result should be kept: {}",
            masked[4].content
        );
        assert!(saved > 0, "should have saved tokens");
    }

    #[test]
    fn masks_oversized_results_regardless_of_age() {
        let counter = EstimateCounter;
        let masker = ResultMasker::new()
            .retain_turns(100)
            .max_result_tokens(10)
            .no_json_truncate();

        let long_output = "x".repeat(500);
        let messages = vec![
            Message::user("Do something"),
            Message::assistant(make_tool_result("shell", &long_output)),
        ];

        let (masked, saved) = masker.mask(&messages, 0, &counter);

        assert!(masked[1].content.contains("masked"));
        assert!(saved > 100);
    }

    #[test]
    fn preserves_non_tool_messages() {
        let counter = EstimateCounter;
        let masker = ResultMasker::new().retain_turns(0);

        let messages = vec![
            Message::system("System prompt"),
            Message::user("Hello"),
            Message::assistant("Hi there!"),
        ];

        let (masked, saved) = masker.mask(&messages, 1, &counter);

        assert_eq!(masked[0].content, "System prompt");
        assert_eq!(masked[1].content, "Hello");
        assert_eq!(masked[2].content, "Hi there!");
        assert_eq!(saved, 0);
    }

    #[test]
    fn respects_retain_turns() {
        let counter = EstimateCounter;
        let masker = ResultMasker::new().retain_turns(5);

        let long_output = "x".repeat(500);
        let messages = vec![
            Message::user("Do something"),
            Message::assistant(make_tool_result("shell", &long_output)),
            Message::user("More"),
            Message::assistant(make_tool_result("shell", &long_output)),
        ];

        let (masked, saved) = masker.mask(&messages, 1, &counter);

        assert!(
            !masked[1].content.contains("masked"),
            "should not mask recent results"
        );
        assert_eq!(saved, 0);
    }

    // ── JSON truncation tests ────────────────────────────────────────────────

    #[test]
    fn truncates_json_string_values() {
        let config = JsonTruncateConfig {
            max_string_len: 20,
            ..Default::default()
        };
        let input = serde_json::json!({
            "short": "hello",
            "long": "a]".repeat(50),
        });
        let result = truncate_json_value(&input, &config, 0);
        let obj = result.as_object().unwrap();

        assert_eq!(obj["short"].as_str().unwrap(), "hello");
        let long_val = obj["long"].as_str().unwrap();
        assert!(long_val.contains("...["), "should be truncated: {long_val}");
        assert!(long_val.contains("chars]"));
    }

    #[test]
    fn truncates_json_arrays() {
        let config = JsonTruncateConfig {
            max_array_items: 2,
            ..Default::default()
        };
        let input = serde_json::json!([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        let result = truncate_json_value(&input, &config, 0);
        let arr = result.as_array().unwrap();

        assert_eq!(arr.len(), 3); // 2 items + summary
        assert_eq!(arr[0], serde_json::json!(1));
        assert_eq!(arr[1], serde_json::json!(2));
        assert_eq!(arr[2].as_str().unwrap(), "...and 8 more items");
    }

    #[test]
    fn truncates_deep_json_nesting() {
        let config = JsonTruncateConfig {
            max_depth: 2,
            ..Default::default()
        };
        let input = serde_json::json!({
            "level1": {
                "level2": {
                    "level3": "deep value"
                }
            }
        });
        let result = truncate_json_value(&input, &config, 0);
        let l1 = &result["level1"];
        let l2 = &l1["level2"];
        assert_eq!(l2.as_str().unwrap(), "[nested object]");
    }

    #[test]
    fn json_truncation_in_masker_preserves_structure() {
        let counter = EstimateCounter;
        let masker = ResultMasker::new()
            .retain_turns(0)
            .json_truncate(JsonTruncateConfig {
                max_string_len: 10,
                max_array_items: 2,
                min_savings_pct: 1.0, // low threshold so truncation is used
                ..Default::default()
            });

        // JSON with long strings — should be truncated, not fully masked
        let json_output = serde_json::json!({
            "stdout": "x".repeat(500),
            "stderr": "",
            "exit_code": 0,
            "files": ["a.rs", "b.rs", "c.rs", "d.rs", "e.rs"]
        })
        .to_string();

        let messages = vec![
            Message::user("Run something"),
            Message::assistant(make_tool_result("shell", &json_output)),
            Message::user("Next"),
        ];

        let (masked, saved) = masker.mask(&messages, 1, &counter);

        // Should contain structural elements (keys preserved)
        assert!(
            masked[1].content.contains("stdout"),
            "JSON keys should be preserved: {}",
            masked[1].content
        );
        assert!(
            masked[1].content.contains("exit_code"),
            "JSON keys should be preserved"
        );
        // Should use JSON truncation, not full masking
        assert!(
            masked[1].content.contains("json truncated"),
            "should use JSON truncation, not full masking: {}",
            masked[1].content
        );
        assert!(saved > 0, "should have saved tokens");
    }

    #[test]
    fn falls_back_to_full_mask_for_non_json() {
        let counter = EstimateCounter;
        let masker = ResultMasker::new().retain_turns(0);

        let plain_output = "line\n".repeat(100);
        let messages = vec![
            Message::user("Run something"),
            Message::assistant(make_tool_result("shell", &plain_output)),
            Message::user("Next"),
        ];

        let (masked, _saved) = masker.mask(&messages, 1, &counter);

        assert!(
            masked[1].content.contains("tokens, masked]"),
            "non-JSON should be fully masked: {}",
            masked[1].content
        );
    }

    #[test]
    fn no_json_truncate_disables_feature() {
        let counter = EstimateCounter;
        let masker = ResultMasker::new().retain_turns(0).no_json_truncate();

        let json_output = serde_json::json!({
            "stdout": "x".repeat(500),
            "exit_code": 0,
        })
        .to_string();

        let messages = vec![
            Message::user("Run something"),
            Message::assistant(make_tool_result("shell", &json_output)),
            Message::user("Next"),
        ];

        let (masked, _saved) = masker.mask(&messages, 1, &counter);

        assert!(
            masked[1].content.contains("tokens, masked]"),
            "should be fully masked when JSON truncation disabled: {}",
            masked[1].content
        );
    }

    // ── Observation/history separation tests ─────────────────────────────────

    #[test]
    fn masks_tool_messages_more_aggressively() {
        let counter = EstimateCounter;
        let masker = ResultMasker::new()
            .retain_turns(3) // base
            .retain_turns_tool(1) // tool results: only keep 1 turn
            .retain_turns_assistant(3) // assistant reasoning: keep 3 turns
            .no_json_truncate();

        let long_output = "x".repeat(500);
        let messages = vec![
            Message::system("System prompt"),
            // Turn 0
            Message::user("Do task A"),
            Message::assistant(make_tool_result("shell", &long_output)),
            Message::tool(long_output.clone()),
            // Turn 1
            Message::user("Do task B"),
            Message::assistant(make_tool_result("shell", &long_output)),
            Message::tool(long_output.clone()),
            // Turn 2
            Message::user("Do task C"),
            Message::assistant(make_tool_result("shell", "recent output")),
            Message::tool("recent tool output"),
            // Turn 3 (current)
            Message::user("What happened?"),
        ];

        // current_turn = 3, tool_retain = 1 → tool_cutoff = 2
        // Turns 0 and 1 tool messages should be masked
        // assistant_retain = 3 → assistant_cutoff = 0, so no assistant masking by age
        let (masked, saved) = masker.mask(&messages, 3, &counter);

        // Tool message from turn 0 should be masked
        assert!(
            masked[3].content.contains("masked"),
            "old Role::Tool message should be masked: {}",
            masked[3].content
        );

        // Tool message from turn 1 should be masked
        assert!(
            masked[6].content.contains("masked"),
            "old Role::Tool message should be masked: {}",
            masked[6].content
        );

        // Tool message from turn 2 should be preserved (within retain_turns_tool=1 from turn 3)
        assert!(
            masked[9].content.contains("recent tool output"),
            "recent Role::Tool message should be kept: {}",
            masked[9].content
        );

        // Assistant messages from turn 0 should NOT be masked (within assistant retain of 3)
        assert!(
            !masked[2].content.contains("tokens, masked]"),
            "assistant reasoning should be preserved: {}",
            masked[2].content
        );

        assert!(saved > 0, "should have saved tokens");
    }

    #[test]
    fn split_retention_backward_compatible() {
        let counter = EstimateCounter;

        // Without split config
        let masker_default = ResultMasker::new().retain_turns(1).no_json_truncate();
        // With split config matching default
        let masker_split = ResultMasker::new()
            .retain_turns(1)
            .retain_turns_tool(1)
            .retain_turns_assistant(1)
            .no_json_truncate();

        let long_output = "x".repeat(500);
        let messages = vec![
            Message::user("Task A"),
            Message::assistant(make_tool_result("shell", &long_output)),
            Message::user("Task B"),
            Message::assistant(make_tool_result("shell", "recent")),
            Message::user("Now?"),
        ];

        let (masked_default, saved_default) = masker_default.mask(&messages, 2, &counter);
        let (masked_split, saved_split) = masker_split.mask(&messages, 2, &counter);

        // Both should produce the same masking behavior
        assert_eq!(
            masked_default[1].content.contains("masked"),
            masked_split[1].content.contains("masked"),
        );
        assert_eq!(saved_default, saved_split);
    }
}
