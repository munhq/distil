use crate::counter::TokenCounter;
use crate::types::{Breakdown, Message, Role, ToolSpec};

/// Analyzes token usage across conversation components.
///
/// Tells you exactly where your tokens are going — system prompt, tool definitions,
/// conversation history, tool results — so you can make informed optimization decisions.
pub struct TokenBudget {
    max_tokens: usize,
}

impl TokenBudget {
    pub fn new(max_tokens: usize) -> Self {
        Self { max_tokens }
    }

    pub fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    /// Analyze token usage in the current context.
    pub fn analyze(
        &self,
        messages: &[Message],
        tools: &[ToolSpec],
        counter: &dyn TokenCounter,
    ) -> Breakdown {
        let mut breakdown = Breakdown::default();

        for msg in messages {
            let tokens = counter.count(&msg.content);
            match msg.role {
                Role::System => breakdown.system += tokens,
                Role::Tool => breakdown.tool_results += tokens,
                _ => breakdown.history += tokens,
            }
        }

        for tool in tools {
            breakdown.tools += counter.count(&tool.to_prompt_text());
        }

        breakdown.total =
            breakdown.system + breakdown.tools + breakdown.history + breakdown.tool_results;
        breakdown
    }

    /// Check if the context fits within budget.
    pub fn fits(&self, breakdown: &Breakdown) -> bool {
        breakdown.total <= self.max_tokens
    }

    /// How many tokens over/under budget.
    pub fn headroom(&self, breakdown: &Breakdown) -> i64 {
        self.max_tokens as i64 - breakdown.total as i64
    }

    /// Trim messages from the oldest non-system messages to fit within budget.
    ///
    /// Returns the trimmed messages and how many tokens were removed.
    /// System messages and the most recent `preserve_recent` messages are never trimmed.
    pub fn trim_to_fit(
        &self,
        messages: &[Message],
        tools: &[ToolSpec],
        counter: &dyn TokenCounter,
        preserve_recent: usize,
    ) -> (Vec<Message>, usize) {
        let tool_tokens: usize = tools
            .iter()
            .map(|t| counter.count(&t.to_prompt_text()))
            .sum();
        let target = self.max_tokens.saturating_sub(tool_tokens);

        // Separate system messages (always kept) from the rest
        let system: Vec<&Message> = messages.iter().filter(|m| m.role == Role::System).collect();
        let non_system: Vec<&Message> =
            messages.iter().filter(|m| m.role != Role::System).collect();

        let system_tokens: usize = system.iter().map(|m| counter.count(&m.content)).sum();
        let available = target.saturating_sub(system_tokens);

        // Count from the end to find how many recent messages we can keep
        let mut kept_tokens = 0;
        let mut keep_from = non_system.len();

        for (i, msg) in non_system.iter().enumerate().rev() {
            let msg_tokens = counter.count(&msg.content);
            if kept_tokens + msg_tokens > available && non_system.len() - i > preserve_recent {
                break;
            }
            kept_tokens += msg_tokens;
            keep_from = i;
        }

        // Calculate savings
        let trimmed_tokens: usize = non_system[..keep_from]
            .iter()
            .map(|m| counter.count(&m.content))
            .sum();

        // Rebuild: system messages first, then kept non-system messages
        let mut result: Vec<Message> = system.into_iter().cloned().collect();
        result.extend(non_system[keep_from..].iter().map(|m| (*m).clone()));

        (result, trimmed_tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::counter::EstimateCounter;

    #[test]
    fn analyze_breaks_down_components() {
        let counter = EstimateCounter;
        let budget = TokenBudget::new(100_000);

        let messages = vec![
            Message::system("You are helpful."),
            Message::user("Hello"),
            Message::assistant("Hi!"),
            Message::tool("command output here"),
        ];

        let tools = vec![ToolSpec {
            name: "shell".into(),
            description: "Run a command".into(),
            parameters: serde_json::json!({"type": "object"}),
        }];

        let breakdown = budget.analyze(&messages, &tools, &counter);

        assert!(breakdown.system > 0);
        assert!(breakdown.history > 0);
        assert!(breakdown.tool_results > 0);
        assert!(breakdown.tools > 0);
        assert_eq!(
            breakdown.total,
            breakdown.system + breakdown.tools + breakdown.history + breakdown.tool_results
        );
    }

    #[test]
    fn fits_within_budget() {
        let budget = TokenBudget::new(1000);
        let breakdown = Breakdown {
            total: 500,
            ..Default::default()
        };
        assert!(budget.fits(&breakdown));

        let over = Breakdown {
            total: 1500,
            ..Default::default()
        };
        assert!(!budget.fits(&over));
    }

    #[test]
    fn trim_removes_oldest_non_system() {
        let counter = EstimateCounter;
        // Very tight budget — must be smaller than total message tokens
        let budget = TokenBudget::new(15);

        let filler = "x".repeat(50); // ~14 tokens each
        let messages = vec![
            Message::system("Sys"),
            Message::user(format!("Old message one {filler}")),
            Message::assistant(format!("Old response one {filler}")),
            Message::user(format!("Old message two {filler}")),
            Message::assistant(format!("Old response two {filler}")),
            Message::user("Recent"),
            Message::assistant("Done"),
        ];

        let (trimmed, saved) = budget.trim_to_fit(&messages, &[], &counter, 2);

        // System message should always be kept
        assert_eq!(trimmed[0].role, Role::System);
        assert_eq!(trimmed[0].content, "Sys");

        // Should have fewer messages than original
        assert!(
            trimmed.len() < messages.len(),
            "should have trimmed: {} vs {}",
            trimmed.len(),
            messages.len()
        );
        assert!(saved > 0);

        // Most recent messages should be preserved
        let last = trimmed.last().unwrap();
        assert_eq!(last.content, "Done");
    }

    #[test]
    fn trim_preserves_all_if_within_budget() {
        let counter = EstimateCounter;
        let budget = TokenBudget::new(100_000);

        let messages = vec![
            Message::system("Sys"),
            Message::user("Hello"),
            Message::assistant("Hi"),
        ];

        let (trimmed, saved) = budget.trim_to_fit(&messages, &[], &counter, 2);

        assert_eq!(trimmed.len(), messages.len());
        assert_eq!(saved, 0);
    }
}
