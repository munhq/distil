use crate::counter::TokenCounter;
use crate::budget::TokenBudget;
use crate::pipeline::{Ctx, Layer, LayerResult};

/// Trims oldest messages to fit within a token budget.
///
/// This is the last-resort layer — run it after registry and masking
/// have already reduced tokens. It removes the oldest non-system messages
/// until the context fits.
///
/// System messages and the N most recent messages are never trimmed.
pub struct BudgetLayer {
    budget: TokenBudget,
    preserve_recent: usize,
}

impl BudgetLayer {
    pub fn new(max_tokens: usize) -> Self {
        Self {
            budget: TokenBudget::new(max_tokens),
            preserve_recent: 6,
        }
    }

    /// Number of most recent messages to always keep (default: 6).
    pub fn preserve_recent(mut self, n: usize) -> Self {
        self.preserve_recent = n;
        self
    }
}

impl Layer for BudgetLayer {
    fn name(&self) -> &str {
        "budget"
    }

    fn apply(&self, ctx: &mut Ctx, counter: &dyn TokenCounter) -> LayerResult {
        let tokens_before = ctx.total_tokens(counter);
        let msg_count_before = ctx.messages.len();

        let (trimmed, _) =
            self.budget
                .trim_to_fit(&ctx.messages, &ctx.tools, counter, self.preserve_recent);
        ctx.messages = trimmed;

        let tokens_after = ctx.total_tokens(counter);
        let removed = msg_count_before - ctx.messages.len();

        LayerResult {
            layer: self.name().into(),
            tokens_before,
            tokens_after,
            detail: format!(
                "budget {}, removed {removed} messages",
                self.budget.max_tokens()
            ),
        }
    }
}
