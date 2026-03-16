//! # Distil — Context optimization for LLM agents
//!
//! Distil is a middleware library that reduces token usage in LLM agent conversations
//! by 50-90%, without degrading task performance.
//!
//! ## Architecture
//!
//! Distil uses a **composable pipeline** of optimization layers. Each layer
//! implements one strategy, runs independently, and reports its own metrics.
//!
//! ```text
//! Agent Loop → Pipeline[Registry → Masking → Compaction → Budget → Cache] → LLM
//! ```
//!
//! ## Layers
//!
//! | Layer | What It Does | Typical Savings |
//! |-------|-------------|----------------|
//! | [`RegistryLayer`] | Compact tool catalog + on-demand loading | 85-95% on tool defs |
//! | [`MaskingLayer`] | Replace old tool outputs with summaries | ~50% on tool results |
//! | [`CompactionLayer`] | Structural dedup, whitespace, merge | 10-30% on bloated history |
//! | [`BudgetLayer`] | Trim oldest messages to fit budget | Prevents overflow |
//! | [`ScratchpadLayer`] | Agent working memory outside context | Survives compaction |
//! | [`CacheAlignLayer`] | Reorder for prompt cache hits | Reduces $/request |
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! let pipeline = Pipeline::builder()
//!     .counter(EstimateCounter)
//!     .layer(RegistryLayer::new(tools, &EstimateCounter))
//!     .layer(MaskingLayer::new().retain_turns(3))
//!     .layer(CompactionLayer::new())
//!     .layer(BudgetLayer::new(32_000))
//!     .layer(CacheAlignLayer::generic())
//!     .build();
//!
//! let mut ctx = Ctx::new(messages, vec![], 5);
//! let result = pipeline.optimize(&mut ctx);
//! println!("{result}");
//! ```

pub mod budget;
#[cfg(feature = "config")]
pub mod config;
pub mod counter;
pub mod error;
pub mod layers;
pub mod masker;
#[cfg(feature = "metrics")]
pub mod metrics;
pub mod pipeline;
pub mod probe;
pub mod registry;
pub mod summarizer;
pub mod types;

// Re-exports for convenience
pub use counter::{counter_for_model, EstimateCounter, TokenCounter, WordCounter};
#[cfg(feature = "tiktoken")]
pub use counter::TiktokenCounter;
pub use error::Error;
pub use layers::*;
pub use masker::JsonTruncateConfig;
pub use pipeline::{Ctx, Layer, LayerResult, OptimizationState, Phase, Pipeline, PipelineResult, ToolExecutor};
pub use probe::{Probe, ProbeEvaluator, ProbeReport, ProbeResult, ProbeType};
pub use summarizer::Summarizer;
#[cfg(feature = "proxy")]
pub use summarizer::{HttpSummarizer, OllamaSummarizer};
pub use types::{Message, ToolSpec};
#[cfg(feature = "config")]
pub use config::PipelineConfig;
#[cfg(feature = "metrics")]
pub use metrics::DistilMetrics;
