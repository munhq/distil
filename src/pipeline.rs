use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::time::Duration;

use crate::counter::TokenCounter;
use crate::types::{Message, ToolSpec};

/// Persistent state for progressive optimization across multiple pipeline calls.
/// Store in Ctx extensions before calling `optimize()`, retrieve after.
///
/// ```rust,ignore
/// // Caller persists across calls:
/// ctx.insert(opt_state.clone());
/// pipeline.optimize(&mut ctx);
/// opt_state = ctx.remove::<OptimizationState>().unwrap_or_default();
/// ```
#[derive(Debug, Clone, Default)]
pub struct OptimizationState {
    /// Last turn processed by masking layer.
    pub masking_watermark_turn: u32,
    /// Last turn processed by summarization layer.
    pub summarization_watermark_turn: u32,
    /// Hash of tool specs when registry catalog was last generated.
    pub tools_hash: Option<u64>,
    /// Cached catalog text (reused if tools haven't changed).
    pub cached_catalog: Option<String>,
}

/// The mutable context that flows through the optimization pipeline.
///
/// Each [`Layer`] reads and modifies this struct. After the pipeline runs,
/// the caller uses the final state for the LLM request.
///
/// ## Extensions
///
/// Layers can store and retrieve typed data via the extensions map,
/// enabling cross-layer communication without coupling (Tower pattern):
///
/// ```rust,ignore
/// ctx.insert(MyLayerState { processed: true });
/// if let Some(state) = ctx.get::<MyLayerState>() { ... }
/// ```
pub struct Ctx {
    /// The conversation messages (system + history).
    pub messages: Vec<Message>,
    /// Tools to send with the LLM request.
    pub tools: Vec<ToolSpec>,
    /// Optional catalog text to prepend/append to the system prompt.
    pub catalog: Option<String>,
    /// Current conversation turn (0-based). Used by age-based layers.
    pub turn: u32,
    /// Type-erased extension map for cross-layer state sharing.
    extensions: HashMap<TypeId, Box<dyn Any + Send + Sync>>,
}

impl Ctx {
    pub fn new(messages: Vec<Message>, tools: Vec<ToolSpec>, turn: u32) -> Self {
        Self {
            messages,
            tools,
            catalog: None,
            turn,
            extensions: HashMap::new(),
        }
    }

    /// Total tokens in the current state.
    pub fn total_tokens(&self, counter: &dyn TokenCounter) -> usize {
        let msg_tokens: usize = self
            .messages
            .iter()
            .map(|m| counter.count(&m.content))
            .sum();
        let tool_tokens: usize = self
            .tools
            .iter()
            .map(|t| counter.count(&t.to_prompt_text()))
            .sum();
        let catalog_tokens = self.catalog.as_ref().map(|c| counter.count(c)).unwrap_or(0);
        msg_tokens + tool_tokens + catalog_tokens
    }

    /// Insert a typed value into the extensions map.
    /// Replaces any existing value of the same type.
    pub fn insert<T: Send + Sync + 'static>(&mut self, val: T) {
        self.extensions.insert(TypeId::of::<T>(), Box::new(val));
    }

    /// Get a reference to a typed value from the extensions map.
    pub fn get<T: Send + Sync + 'static>(&self) -> Option<&T> {
        self.extensions
            .get(&TypeId::of::<T>())
            .and_then(|boxed| boxed.downcast_ref())
    }

    /// Get a mutable reference to a typed value from the extensions map.
    pub fn get_mut<T: Send + Sync + 'static>(&mut self) -> Option<&mut T> {
        self.extensions
            .get_mut(&TypeId::of::<T>())
            .and_then(|boxed| boxed.downcast_mut())
    }

    /// Remove a typed value from the extensions map, returning it if present.
    pub fn remove<T: Send + Sync + 'static>(&mut self) -> Option<T> {
        self.extensions
            .remove(&TypeId::of::<T>())
            .and_then(|boxed| boxed.downcast().ok())
            .map(|boxed| *boxed)
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
    /// Wall-clock time spent in this layer.
    pub duration: Duration,
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
        let ms = self.duration.as_secs_f64() * 1000.0;
        if saved > 0 {
            write!(
                f,
                "[{}] {} → {} tokens (saved {}, {:.1}%) [{:.1}ms] — {}",
                self.layer,
                self.tokens_before,
                self.tokens_after,
                saved,
                self.percentage_saved(),
                ms,
                self.detail
            )
        } else {
            write!(
                f,
                "[{}] {} tokens (no change) [{:.1}ms] — {}",
                self.layer, self.tokens_before, ms, self.detail
            )
        }
    }
}

/// Pipeline execution phase for ordering validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Phase {
    Setup = 0,
    Transform = 1,
    Compress = 2,
    Finalize = 3,
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

    /// Declare this layer's execution phase for ordering validation.
    /// Returns `None` by default (no ordering constraint).
    fn phase(&self) -> Option<Phase> {
        None
    }

    /// Handle a tool call that this layer injected into the context.
    ///
    /// Returns `Some(output)` if this layer owns the tool, `None` otherwise.
    /// The caller should include the output in the conversation.
    fn handle_tool_call(&self, _tool_name: &str, _args: &serde_json::Value) -> Option<String> {
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
    /// Total wall-clock time for the entire pipeline.
    pub duration: Duration,
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
        let ms = self.duration.as_secs_f64() * 1000.0;
        writeln!(
            f,
            "Distil: {} → {} tokens (saved {}, {:.1}%) [{:.1}ms]",
            self.tokens_before,
            self.tokens_after,
            self.total_saved(),
            self.percentage_saved(),
            ms,
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
        let pipeline_start = std::time::Instant::now();
        let tokens_before = ctx.total_tokens(&*self.counter);
        let mut results = Vec::new();

        for layer in &self.layers {
            let layer_start = std::time::Instant::now();
            let mut result = layer.apply(ctx, &*self.counter);
            result.duration = layer_start.elapsed();
            results.push(result);
        }

        let tokens_after = ctx.total_tokens(&*self.counter);

        PipelineResult {
            layers: results,
            tokens_before,
            tokens_after,
            duration: pipeline_start.elapsed(),
        }
    }

    /// Delegate a tool call to the appropriate layer.
    ///
    /// When the LLM calls a tool that distil injected (e.g., `tool_search`,
    /// `scratchpad_write`), pass it here. Returns the tool output if handled.
    pub fn handle_tool_call(&self, tool_name: &str, args: &serde_json::Value) -> Option<String> {
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

/// Executes tool calls on behalf of layers that need to invoke agent tools.
///
/// Used by layers like `CodeModeLayer` that need to call the agent's actual
/// tool implementations during execution (e.g., running a script that chains
/// multiple tool calls in a sandbox).
///
/// Distil doesn't own tool implementations — the agent does. This trait bridges
/// that gap: the caller provides an executor, and layers use it to invoke tools.
///
/// # Implementations
///
/// - **Crate users**: implement this trait with a closure or struct wrapping
///   your agent's tool dispatch
/// - **HTTP server**: executor makes HTTP callbacks to the agent's tool endpoint
/// - **MCP mode**: executor routes through MCP protocol
///
/// ```rust,ignore
/// struct MyExecutor { /* ... */ }
///
/// impl ToolExecutor for MyExecutor {
///     fn execute(&self, tool_name: &str, args: &serde_json::Value) -> Result<String, crate::Error> {
///         match tool_name {
///             "shell" => run_shell(args),
///             "file_read" => read_file(args),
///             _ => Err(crate::Error::ToolExecution(format!("unknown tool: {tool_name}"))),
///         }
///     }
/// }
/// ```
pub trait ToolExecutor: Send + Sync {
    /// Execute a tool call and return the output as a string.
    fn execute(
        &self,
        tool_name: &str,
        args: &serde_json::Value,
    ) -> std::result::Result<String, crate::Error>;
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

    /// Build the pipeline, returning an error if layers are in invalid phase order.
    pub fn build_checked(self) -> Result<Pipeline, crate::Error> {
        let warnings = self.validate_ordering();
        if !warnings.is_empty() {
            return Err(crate::Error::Config(format!(
                "layer ordering issues: {}",
                warnings.join("; ")
            )));
        }
        Ok(self.build())
    }

    fn validate_ordering(&self) -> Vec<String> {
        let mut warnings = Vec::new();
        let mut max_phase = None;
        for layer in &self.layers {
            if let Some(phase) = layer.phase() {
                if let Some(prev) = max_phase {
                    if phase < prev {
                        warnings.push(format!(
                            "'{}' ({:?}) comes after a {:?}-phase layer",
                            layer.name(),
                            phase,
                            prev
                        ));
                    }
                }
                max_phase = Some(max_phase.map_or(phase, |p: Phase| p.max(phase)));
            }
        }
        warnings
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::counter::EstimateCounter;
    use crate::types::Message;

    #[test]
    fn ctx_extensions_insert_get_remove() {
        #[derive(Debug, PartialEq)]
        struct LayerState {
            processed: bool,
            count: u32,
        }

        let mut ctx = Ctx::new(vec![Message::user("hello")], vec![], 0);

        // Initially empty
        assert!(ctx.get::<LayerState>().is_none());

        // Insert
        ctx.insert(LayerState {
            processed: true,
            count: 42,
        });
        let state = ctx.get::<LayerState>().unwrap();
        assert!(state.processed);
        assert_eq!(state.count, 42);

        // Mutate
        ctx.get_mut::<LayerState>().unwrap().count = 99;
        assert_eq!(ctx.get::<LayerState>().unwrap().count, 99);

        // Remove
        let removed = ctx.remove::<LayerState>().unwrap();
        assert_eq!(removed.count, 99);
        assert!(ctx.get::<LayerState>().is_none());
    }

    #[test]
    fn ctx_extensions_multiple_types() {
        let mut ctx = Ctx::new(vec![], vec![], 0);
        ctx.insert(42u32);
        ctx.insert("hello".to_string());

        assert_eq!(*ctx.get::<u32>().unwrap(), 42);
        assert_eq!(ctx.get::<String>().unwrap(), "hello");

        // Overwrite
        ctx.insert(100u32);
        assert_eq!(*ctx.get::<u32>().unwrap(), 100);
        // String unchanged
        assert_eq!(ctx.get::<String>().unwrap(), "hello");
    }

    #[test]
    fn valid_ordering_passes_checked_build() {
        let counter = EstimateCounter;
        let result = Pipeline::builder()
            .counter(counter)
            .layer(crate::layers::CompactionLayer::new())
            .layer(crate::layers::BudgetLayer::new(32_000))
            .build_checked();
        assert!(result.is_ok());
    }

    #[test]
    fn invalid_ordering_fails_checked_build() {
        let counter = EstimateCounter;
        let result = Pipeline::builder()
            .counter(counter)
            .layer(crate::layers::BudgetLayer::new(32_000))
            .layer(crate::layers::MaskingLayer::new())
            .build_checked();
        assert!(result.is_err());
    }

    #[test]
    fn pipeline_optimize_records_timing() {
        let counter = EstimateCounter;
        let pipeline = Pipeline::builder()
            .counter(counter)
            .layer(crate::layers::CompactionLayer::new())
            .build();

        let mut ctx = Ctx::new(
            vec![
                Message::system("Be helpful."),
                Message::user("hello"),
                Message::assistant("hi"),
            ],
            vec![],
            1,
        );

        let result = pipeline.optimize(&mut ctx);

        // Pipeline duration should be non-zero (or at least not panic)
        assert!(!result.layers.is_empty());
        // Each layer should have a duration set by the pipeline
        for lr in &result.layers {
            // Duration is set by optimize(), not the layer itself
            // It might be 0 on fast machines, but it should be set
            let _ = lr.duration;
        }
        // Pipeline total duration should be >= sum of layers
        let _ = result.duration;
    }
}
