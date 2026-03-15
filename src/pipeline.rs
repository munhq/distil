use crate::counter::TokenCounter;
use crate::types::{Message, ToolSpec};

/// The mutable context that flows through the optimization pipeline.
///
/// Each [`Layer`] reads and modifies this struct. After the pipeline runs,
/// the caller uses the final state for the LLM request.
pub struct Ctx {
    /// The conversation messages (system + history).
    pub messages: Vec<Message>,
    /// Tools to send with the LLM request.
    pub tools: Vec<ToolSpec>,
    /// Optional catalog text to prepend/append to the system prompt.
    pub catalog: Option<String>,
    /// Current conversation turn (0-based). Used by age-based layers.
    pub turn: u32,
}

impl Ctx {
    pub fn new(messages: Vec<Message>, tools: Vec<ToolSpec>, turn: u32) -> Self {
        Self {
            messages,
            tools,
            catalog: None,
            turn,
        }
    }

    /// Total tokens in the current state.
    pub fn total_tokens(&self, counter: &dyn TokenCounter) -> usize {
        let msg_tokens: usize = self.messages.iter().map(|m| counter.count(&m.content)).sum();
        let tool_tokens: usize = self
            .tools
            .iter()
            .map(|t| counter.count(&t.to_prompt_text()))
            .sum();
        let catalog_tokens = self
            .catalog
            .as_ref()
            .map(|c| counter.count(c))
            .unwrap_or(0);
        msg_tokens + tool_tokens + catalog_tokens
    }
}

/// Result from a single layer's optimization pass.
#[derive(Debug, Clone)]
pub struct LayerResult {
    /// Layer name (for reporting).
    pub layer: String,
    /// Tokens in the context before this layer ran.
    pub tokens_before: usize,
    /// Tokens in the context after this layer ran.
    pub tokens_after: usize,
    /// Human-readable detail about what changed.
    pub detail: String,
}

impl LayerResult {
    pub fn tokens_saved(&self) -> usize {
        self.tokens_before.saturating_sub(self.tokens_after)
    }

    pub fn percentage_saved(&self) -> f64 {
        if self.tokens_before == 0 {
            return 0.0;
        }
        (self.tokens_saved() as f64 / self.tokens_before as f64) * 100.0
    }
}

impl std::fmt::Display for LayerResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let saved = self.tokens_saved();
        if saved > 0 {
            write!(
                f,
                "[{}] {} → {} tokens (saved {}, {:.1}%) — {}",
                self.layer,
                self.tokens_before,
                self.tokens_after,
                saved,
                self.percentage_saved(),
                self.detail
            )
        } else {
            write!(f, "[{}] {} tokens (no change) — {}", self.layer, self.tokens_before, self.detail)
        }
    }
}

/// A composable optimization layer.
///
/// Each layer implements one specific optimization strategy. Layers are
/// run in sequence by the [`Pipeline`], each modifying the [`Ctx`] in place.
///
/// Layers are sync by design — distil is LLM-agnostic and never calls an
/// LLM itself. If a layer needs LLM output (e.g., summarization), the caller
/// provides it upfront.
pub trait Layer: Send + Sync {
    /// Layer name for metrics and logging.
    fn name(&self) -> &str;

    /// Apply this optimization to the context. Returns metrics about what changed.
    fn apply(&self, ctx: &mut Ctx, counter: &dyn TokenCounter) -> LayerResult;

    /// Handle a tool call that this layer injected into the context.
    ///
    /// Returns `Some(output)` if this layer owns the tool, `None` otherwise.
    /// The caller should include the output in the conversation.
    fn handle_tool_call(
        &self,
        _tool_name: &str,
        _args: &serde_json::Value,
    ) -> Option<String> {
        None
    }
}

/// Full result from running the optimization pipeline.
#[derive(Debug)]
pub struct PipelineResult {
    /// Per-layer results, in order of execution.
    pub layers: Vec<LayerResult>,
    /// Total tokens before any optimization.
    pub tokens_before: usize,
    /// Total tokens after all optimizations.
    pub tokens_after: usize,
}

impl PipelineResult {
    pub fn total_saved(&self) -> usize {
        self.tokens_before.saturating_sub(self.tokens_after)
    }

    pub fn percentage_saved(&self) -> f64 {
        if self.tokens_before == 0 {
            return 0.0;
        }
        (self.total_saved() as f64 / self.tokens_before as f64) * 100.0
    }
}

impl std::fmt::Display for PipelineResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "Distil: {} → {} tokens (saved {}, {:.1}%)",
            self.tokens_before,
            self.tokens_after,
            self.total_saved(),
            self.percentage_saved()
        )?;
        for layer in &self.layers {
            writeln!(f, "  {layer}")?;
        }
        Ok(())
    }
}

/// The optimization pipeline. Composes layers in sequence.
///
/// ```rust,ignore
/// let pipeline = Pipeline::builder()
///     .counter(EstimateCounter)
///     .layer(RegistryLayer::new(tools, &EstimateCounter))
///     .layer(MaskingLayer::new().retain_turns(3))
///     .layer(BudgetLayer::new(32_000).preserve_recent(6))
///     .build();
///
/// let mut ctx = Ctx::new(messages, vec![], 5);
/// let result = pipeline.optimize(&mut ctx);
/// ```
pub struct Pipeline {
    layers: Vec<Box<dyn Layer>>,
    counter: Box<dyn TokenCounter>,
}

impl Pipeline {
    pub fn builder() -> PipelineBuilder {
        PipelineBuilder {
            layers: Vec::new(),
            counter: None,
        }
    }

    /// Run all layers in sequence.
    pub fn optimize(&self, ctx: &mut Ctx) -> PipelineResult {
        let tokens_before = ctx.total_tokens(&*self.counter);
        let mut results = Vec::new();

        for layer in &self.layers {
            let before = ctx.total_tokens(&*self.counter);
            let result = layer.apply(ctx, &*self.counter);
            let _ = before; // result already has before/after
            results.push(result);
        }

        let tokens_after = ctx.total_tokens(&*self.counter);

        PipelineResult {
            layers: results,
            tokens_before,
            tokens_after,
        }
    }

    /// Delegate a tool call to the appropriate layer.
    ///
    /// When the LLM calls a tool that distil injected (e.g., `tool_search`,
    /// `scratchpad_write`), pass it here. Returns the tool output if handled.
    pub fn handle_tool_call(
        &self,
        tool_name: &str,
        args: &serde_json::Value,
    ) -> Option<String> {
        for layer in &self.layers {
            if let Some(result) = layer.handle_tool_call(tool_name, args) {
                return Some(result);
            }
        }
        None
    }

    /// Access the token counter.
    pub fn counter(&self) -> &dyn TokenCounter {
        &*self.counter
    }
}

pub struct PipelineBuilder {
    layers: Vec<Box<dyn Layer>>,
    counter: Option<Box<dyn TokenCounter>>,
}

impl PipelineBuilder {
    /// Set the token counter (default: EstimateCounter).
    pub fn counter(mut self, counter: impl TokenCounter + 'static) -> Self {
        self.counter = Some(Box::new(counter));
        self
    }

    /// Add a layer to the pipeline.
    pub fn layer(mut self, layer: impl Layer + 'static) -> Self {
        self.layers.push(Box::new(layer));
        self
    }

    pub fn build(self) -> Pipeline {
        Pipeline {
            layers: self.layers,
            counter: self
                .counter
                .unwrap_or_else(|| Box::new(crate::counter::EstimateCounter)),
        }
    }
}
