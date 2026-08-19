//! TOML-based pipeline configuration.
//!
//! Enables declarative pipeline definition without code. Parse a TOML file or
//! string into a [`PipelineConfig`], then build a [`Pipeline`] from it.
//!
//! # Example
//!
//! ```toml
//! [pipeline]
//! budget = 32000
//! model = "gpt-4o"
//!
//! [[layers]]
//! type = "registry"
//!
//! [[layers]]
//! type = "masking"
//! retain_turns = 3
//! retain_turns_tool = 1
//! retain_turns_assistant = 4
//! json_truncate = true
//!
//! [[layers]]
//! type = "compactor"
//!
//! [[layers]]
//! type = "budget"
//! max_tokens = 32000
//! preserve_recent = 6
//!
//! [[layers]]
//! type = "scratchpad"
//! max_entries = 50
//!
//! [[layers]]
//! type = "cache_align"
//! provider = "anthropic"
//! ```
//!
//! ```rust,ignore
//! let config = PipelineConfig::parse_str(toml_str)?;
//! let pipeline = config.build_pipeline(&tools, None::<NoSummarizer>, None);
//! ```

use serde::Deserialize;

use crate::counter;
use crate::error::Error;
use crate::layers::*;
use crate::pipeline::Pipeline;
use crate::types::ToolSpec;

/// Top-level configuration parsed from TOML.
#[derive(Debug, Deserialize)]
pub struct PipelineConfig {
    /// Global pipeline settings.
    #[serde(default)]
    pub pipeline: PipelineSettings,
    /// Ordered list of layer configurations.
    #[serde(default)]
    pub layers: Vec<LayerConfig>,
}

/// Global pipeline settings.
#[derive(Debug, Deserialize)]
pub struct PipelineSettings {
    /// Token budget (default: 32000).
    #[serde(default = "default_budget")]
    pub budget: usize,
    /// Model name for token counting (default: "gpt-4o").
    #[serde(default = "default_model")]
    pub model: String,
}

impl Default for PipelineSettings {
    fn default() -> Self {
        Self {
            budget: default_budget(),
            model: default_model(),
        }
    }
}

fn default_budget() -> usize {
    32_000
}

fn default_model() -> String {
    "gpt-4o".into()
}

/// Configuration for a single layer.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum LayerConfig {
    Registry,
    Masking(MaskingConfig),
    Summarization(SummarizationConfig),
    Compactor,
    Budget(BudgetConfig),
    Scratchpad(ScratchpadConfig),
    CacheAlign(CacheAlignConfig),
    #[cfg(feature = "code-mode")]
    CodeMode(CodeModeConfig),
}

/// MaskingLayer configuration.
#[derive(Debug, Default, Deserialize)]
pub struct MaskingConfig {
    pub retain_turns: Option<u32>,
    pub retain_turns_tool: Option<u32>,
    pub retain_turns_assistant: Option<u32>,
    pub max_result_tokens: Option<usize>,
    #[serde(default = "default_true")]
    pub json_truncate: bool,
}

/// SummarizationLayer configuration.
#[derive(Debug, Default, Deserialize)]
pub struct SummarizationConfig {
    pub age_threshold: Option<u32>,
    pub max_summary_tokens: Option<usize>,
    pub min_content_tokens: Option<usize>,
}

/// BudgetLayer configuration.
#[derive(Debug, Default, Deserialize)]
pub struct BudgetConfig {
    pub max_tokens: Option<usize>,
    pub preserve_recent: Option<usize>,
}

/// ScratchpadLayer configuration.
#[derive(Debug, Default, Deserialize)]
pub struct ScratchpadConfig {
    pub max_entries: Option<usize>,
    pub max_value_tokens: Option<usize>,
}

/// CacheAlignLayer configuration.
#[derive(Debug, Default, Deserialize)]
pub struct CacheAlignConfig {
    #[serde(default = "default_provider")]
    pub provider: String,
}

/// CodeModeLayer configuration.
#[cfg(feature = "code-mode")]
#[derive(Debug, Default, Deserialize)]
pub struct CodeModeConfig {
    #[serde(default = "default_timeout_seconds")]
    pub timeout_seconds: u64,
    #[serde(default = "default_memory_limit_mb")]
    pub memory_limit_mb: usize,
    #[serde(default)]
    pub tool_names: Vec<String>,
}

#[cfg(feature = "code-mode")]
fn default_timeout_seconds() -> u64 {
    10
}
#[cfg(feature = "code-mode")]
fn default_memory_limit_mb() -> usize {
    256
}

fn default_true() -> bool {
    true
}

fn default_provider() -> String {
    "generic".into()
}

impl std::str::FromStr for PipelineConfig {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        toml::from_str(s).map_err(|e| Error::Config(e.to_string()))
    }
}

impl PipelineConfig {
    /// Parse a TOML string into a pipeline configuration.
    pub fn parse_str(toml_str: &str) -> Result<Self, Error> {
        toml_str.parse()
    }

    /// Parse a TOML file into a pipeline configuration.
    pub fn from_file(path: &str) -> Result<Self, Error> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| Error::Config(format!("failed to read {path}: {e}")))?;
        content.parse()
    }

    /// Build a [`Pipeline`] from this configuration.
    ///
    /// - `tools` — tool specs for RegistryLayer (pass `&[]` if not using registry)
    /// - `summarizer` — optional LLM summarizer; if `None` and a summarization
    ///   layer is configured, that layer is skipped
    /// - `executor` — optional tool executor for CodeModeLayer; if `None` and a
    ///   code_mode layer is configured, that layer is skipped
    pub fn build_pipeline<S: crate::summarizer::Summarizer + 'static>(
        &self,
        tools: &[ToolSpec],
        summarizer: Option<S>,
        executor: Option<std::sync::Arc<dyn crate::pipeline::ToolExecutor>>,
    ) -> Pipeline {
        // Suppress unused warning when code-mode feature is disabled
        let _ = &executor;

        let counter = counter::counter_for_model(&self.pipeline.model);
        let mut builder =
            Pipeline::builder().counter(counter::counter_for_model(&self.pipeline.model));

        // Summarizer is consumed at most once
        let mut summarizer = summarizer;

        for layer_config in &self.layers {
            match layer_config {
                LayerConfig::Registry => {
                    if !tools.is_empty() {
                        builder = builder.layer(RegistryLayer::new(tools.to_vec(), &*counter));
                    }
                }
                LayerConfig::Masking(cfg) => {
                    let mut layer = MaskingLayer::new();
                    if let Some(rt) = cfg.retain_turns {
                        layer = layer.retain_turns(rt);
                    }
                    if let Some(rt) = cfg.retain_turns_tool {
                        layer = layer.retain_turns_tool(rt);
                    }
                    if let Some(ra) = cfg.retain_turns_assistant {
                        layer = layer.retain_turns_assistant(ra);
                    }
                    if let Some(max) = cfg.max_result_tokens {
                        layer = layer.max_result_tokens(max);
                    }
                    if !cfg.json_truncate {
                        layer = layer.no_json_truncate();
                    }
                    builder = builder.layer(layer);
                }
                LayerConfig::Summarization(cfg) => {
                    if let Some(s) = summarizer.take() {
                        let mut layer = SummarizationLayer::new(s);
                        if let Some(age) = cfg.age_threshold {
                            layer = layer.age_threshold(age);
                        }
                        if let Some(max) = cfg.max_summary_tokens {
                            layer = layer.max_summary_tokens(max);
                        }
                        if let Some(min) = cfg.min_content_tokens {
                            layer = layer.min_content_tokens(min);
                        }
                        builder = builder.layer(layer);
                    }
                    // If no summarizer provided, skip silently
                }
                LayerConfig::Compactor => {
                    builder = builder.layer(CompactionLayer::new());
                }
                LayerConfig::Budget(cfg) => {
                    let max = cfg.max_tokens.unwrap_or(self.pipeline.budget);
                    let mut layer = BudgetLayer::new(max);
                    if let Some(pr) = cfg.preserve_recent {
                        layer = layer.preserve_recent(pr);
                    }
                    builder = builder.layer(layer);
                }
                LayerConfig::Scratchpad(cfg) => {
                    let mut layer = ScratchpadLayer::new();
                    if let Some(max) = cfg.max_entries {
                        layer = layer.max_entries(max);
                    }
                    if let Some(max) = cfg.max_value_tokens {
                        layer = layer.max_value_tokens(max);
                    }
                    builder = builder.layer(layer);
                }
                LayerConfig::CacheAlign(cfg) => {
                    let layer = match cfg.provider.as_str() {
                        "anthropic" => CacheAlignLayer::anthropic(),
                        "openai" => CacheAlignLayer::openai(),
                        _ => CacheAlignLayer::generic(),
                    };
                    builder = builder.layer(layer);
                }
                #[cfg(feature = "code-mode")]
                LayerConfig::CodeMode(cfg) => {
                    if let Some(ref exec) = executor {
                        let mut layer = CodeModeLayer::from_arc(exec.clone())
                            .timeout(std::time::Duration::from_secs(cfg.timeout_seconds));
                        if cfg.memory_limit_mb > 0 {
                            layer = layer.memory_limit(cfg.memory_limit_mb * 1024 * 1024);
                        }
                        if !cfg.tool_names.is_empty() {
                            layer = layer.tool_names(cfg.tool_names.clone());
                        }
                        builder = builder.layer(layer);
                    }
                    // If no executor provided, skip silently
                }
            }
        }

        builder.build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::Ctx;
    use crate::types::Message;

    #[test]
    fn parses_minimal_config() {
        let config = PipelineConfig::parse_str(
            r#"
            [[layers]]
            type = "compactor"
            "#,
        )
        .unwrap();

        assert_eq!(config.pipeline.budget, 32_000);
        assert_eq!(config.pipeline.model, "gpt-4o");
        assert_eq!(config.layers.len(), 1);
    }

    #[test]
    fn parses_full_config() {
        let config = PipelineConfig::parse_str(
            r#"
            [pipeline]
            budget = 16000
            model = "claude-sonnet-4-20250514"

            [[layers]]
            type = "registry"

            [[layers]]
            type = "masking"
            retain_turns = 2
            retain_turns_tool = 1
            retain_turns_assistant = 3
            json_truncate = true

            [[layers]]
            type = "summarization"
            age_threshold = 3
            max_summary_tokens = 150
            min_content_tokens = 50

            [[layers]]
            type = "compactor"

            [[layers]]
            type = "budget"
            max_tokens = 16000
            preserve_recent = 4

            [[layers]]
            type = "scratchpad"
            max_entries = 30
            max_value_tokens = 300

            [[layers]]
            type = "cache_align"
            provider = "anthropic"
            "#,
        )
        .unwrap();

        assert_eq!(config.pipeline.budget, 16_000);
        assert_eq!(config.pipeline.model, "claude-sonnet-4-20250514");
        assert_eq!(config.layers.len(), 7);

        // Verify masking config
        match &config.layers[1] {
            LayerConfig::Masking(cfg) => {
                assert_eq!(cfg.retain_turns, Some(2));
                assert_eq!(cfg.retain_turns_tool, Some(1));
                assert!(cfg.json_truncate);
            }
            _ => panic!("expected masking layer"),
        }

        // Verify cache_align config
        match &config.layers[6] {
            LayerConfig::CacheAlign(cfg) => {
                assert_eq!(cfg.provider, "anthropic");
            }
            _ => panic!("expected cache_align layer"),
        }
    }

    #[test]
    fn builds_pipeline_from_config() {
        let config = PipelineConfig::parse_str(
            r#"
            [[layers]]
            type = "registry"

            [[layers]]
            type = "masking"
            retain_turns = 2

            [[layers]]
            type = "compactor"

            [[layers]]
            type = "budget"
            preserve_recent = 4

            [[layers]]
            type = "cache_align"
            provider = "generic"
            "#,
        )
        .unwrap();

        let tools = vec![ToolSpec {
            name: "shell".into(),
            description: "Run a command".into(),
            parameters: serde_json::json!({"type": "object"}),
        }];

        let pipeline = config.build_pipeline(&tools, None::<MockSummarizer>, None);

        let mut ctx = Ctx::new(
            vec![
                Message::system("Be helpful."),
                Message::user("Hello"),
                Message::assistant("Hi there!"),
            ],
            tools,
            1,
        );

        let result = pipeline.optimize(&mut ctx);
        // Should not panic and should produce results
        assert!(!result.layers.is_empty());
    }

    #[test]
    fn builds_pipeline_with_summarizer() {
        let config = PipelineConfig::parse_str(
            r#"
            [[layers]]
            type = "summarization"
            age_threshold = 1
            min_content_tokens = 5
            "#,
        )
        .unwrap();

        let summarizer = MockSummarizer;
        let pipeline = config.build_pipeline::<MockSummarizer>(&[], Some(summarizer), None);

        let mut ctx = Ctx::new(
            vec![
                Message::system("System"),
                Message::user("Build auth"),
                Message::assistant("Done building auth. ".repeat(20)),
                Message::user("Run tests"),
                Message::assistant("All tests pass. ".repeat(10)),
                Message::user("Next?"),
            ],
            vec![],
            2,
        );

        let result = pipeline.optimize(&mut ctx);
        assert!(
            result.layers.iter().any(|l| l.layer == "summarization"),
            "summarization layer should be present"
        );
    }

    #[test]
    fn skips_summarization_without_summarizer() {
        let config = PipelineConfig::parse_str(
            r#"
            [[layers]]
            type = "summarization"

            [[layers]]
            type = "compactor"
            "#,
        )
        .unwrap();

        // Pass None — summarization should be skipped, compactor still runs
        let pipeline = config.build_pipeline::<MockSummarizer>(&[], None, None);

        let mut ctx = Ctx::new(vec![Message::user("Hello")], vec![], 0);
        let result = pipeline.optimize(&mut ctx);

        // Only compactor should be present (summarization skipped)
        assert_eq!(result.layers.len(), 1);
        assert_eq!(result.layers[0].layer, "compactor");
    }

    #[test]
    fn rejects_invalid_toml() {
        let result = PipelineConfig::parse_str("[[layers]]\ntype = invalid");
        assert!(result.is_err());
    }

    #[test]
    fn rejects_unknown_layer_type() {
        let result = PipelineConfig::parse_str(
            r#"
            [[layers]]
            type = "nonexistent"
            "#,
        );
        assert!(result.is_err());
    }

    struct MockSummarizer;
    impl crate::summarizer::Summarizer for MockSummarizer {
        fn summarize(&self, _content: &str, _max_tokens: usize) -> crate::error::Result<String> {
            Ok("Summary of old conversation.".into())
        }
    }
}
