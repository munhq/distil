use crate::counter::TokenCounter;
use crate::masker::{JsonTruncateConfig, ResultMasker};
use crate::pipeline::{Ctx, Layer, LayerResult};

/// Replaces old tool outputs with compact one-line summaries.
///
/// Tool results are typically the single largest token consumer in agent
/// conversations. A `shell` command can produce hundreds of lines the LLM
/// saw once and doesn't need again.
///
/// This layer replaces old results with: `[shell → 2,412 tokens, masked]`
///
/// When tool results contain valid JSON, the layer can preserve the structure
/// (keys, types) while truncating long values — see [`JsonTruncateConfig`].
///
/// Supports separate retention policies for tool results (observations) vs
/// assistant reasoning (history) via [`retain_turns_tool`] and
/// [`retain_turns_assistant`].
///
/// Based on JetBrains "Complexity Trap" research (NeurIPS 2025) showing
/// ~50% savings with zero quality degradation.
pub struct MaskingLayer {
    masker: ResultMasker,
}

impl MaskingLayer {
    pub fn new() -> Self {
        Self {
            masker: ResultMasker::new(),
        }
    }

    /// Mask results older than N turns (default: 3).
    pub fn retain_turns(mut self, turns: u32) -> Self {
        self.masker = self.masker.retain_turns(turns);
        self
    }

    /// Set separate retention for `Role::Tool` messages (observations).
    /// Tool results are pure data and can be compressed more aggressively.
    pub fn retain_turns_tool(mut self, turns: u32) -> Self {
        self.masker = self.masker.retain_turns_tool(turns);
        self
    }

    /// Set separate retention for `Role::Assistant` messages with embedded tool results.
    /// Assistant reasoning is needed for the LLM to follow its own logic.
    pub fn retain_turns_assistant(mut self, turns: u32) -> Self {
        self.masker = self.masker.retain_turns_assistant(turns);
        self
    }

    /// Mask any single result over N tokens, regardless of age.
    pub fn max_result_tokens(mut self, max: usize) -> Self {
        self.masker = self.masker.max_result_tokens(max);
        self
    }

    /// Set JSON truncation config.
    pub fn json_truncate(mut self, config: JsonTruncateConfig) -> Self {
        self.masker = self.masker.json_truncate(config);
        self
    }

    /// Disable JSON truncation — always fully mask.
    pub fn no_json_truncate(mut self) -> Self {
        self.masker = self.masker.no_json_truncate();
        self
    }
}

impl Default for MaskingLayer {
    fn default() -> Self {
        Self::new()
    }
}

impl Layer for MaskingLayer {
    fn name(&self) -> &str {
        "masking"
    }

    fn apply(&self, ctx: &mut Ctx, counter: &dyn TokenCounter) -> LayerResult {
        let tokens_before = ctx.total_tokens(counter);

        let (masked, tokens_saved) = self.masker.mask(&ctx.messages, ctx.turn, counter);
        ctx.messages = masked;

        let tokens_after = ctx.total_tokens(counter);

        LayerResult {
            layer: self.name().into(),
            tokens_before,
            tokens_after,
            detail: format!("masked old tool results, saved {tokens_saved} tokens"),
        }
    }
}
