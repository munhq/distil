use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::time::Duration;

use crate::counter::TokenCounter;
use crate::pipeline::{Ctx, Layer, LayerResult, OptimizationState};
use crate::registry::ToolRegistry;
use crate::types::ToolSpec;

fn hash_tools(tools: &[ToolSpec]) -> u64 {
    let mut hasher = DefaultHasher::new();
    for tool in tools {
        tool.name.hash(&mut hasher);
        tool.description.hash(&mut hasher);
    }
    hasher.finish()
}

/// Replaces full tool schemas with a compact catalog + `tool_search` meta-tool.
///
/// Instead of injecting all tool schemas (~10K+ tokens for 40+ tools), this layer:
/// 1. Removes all tools from `ctx.tools`
/// 2. Adds only the `tool_search` meta-tool
/// 3. Sets `ctx.catalog` to a compact listing (name + one-line per tool)
///
/// When the LLM needs a tool, it calls `tool_search("query")` and gets full schemas
/// for matching tools. Pass `tool_search` calls to [`handle_tool_call`].
///
/// Typical savings: **85-95%** on tool definition tokens.
pub struct RegistryLayer {
    registry: ToolRegistry,
}

impl RegistryLayer {
    pub fn new(tools: Vec<ToolSpec>, counter: &dyn TokenCounter) -> Self {
        Self {
            registry: ToolRegistry::new(tools, counter),
        }
    }

    /// Search tools by keyword. Used internally by `handle_tool_call`.
    pub fn search(&self, query: &str, max_results: usize) -> Vec<&ToolSpec> {
        self.registry.search(query, max_results)
    }

    /// Get a tool by exact name.
    pub fn get(&self, name: &str) -> Option<&ToolSpec> {
        self.registry.get(name)
    }

    /// Access the underlying registry.
    pub fn registry(&self) -> &ToolRegistry {
        &self.registry
    }
}

impl Layer for RegistryLayer {
    fn name(&self) -> &str {
        "registry"
    }

    fn phase(&self) -> Option<crate::pipeline::Phase> {
        Some(crate::pipeline::Phase::Setup)
    }

    fn apply(&self, ctx: &mut Ctx, counter: &dyn TokenCounter) -> LayerResult {
        let tokens_before = ctx.total_tokens(counter);
        let current_hash = hash_tools(&ctx.tools);

        // Check if we can reuse cached catalog from OptimizationState
        if let Some(state) = ctx.get::<OptimizationState>() {
            if state.tools_hash == Some(current_hash) {
                if let Some(ref cached) = state.cached_catalog {
                    ctx.catalog = Some(cached.clone());
                    ctx.tools = vec![self.registry.search_tool_spec()];
                    #[cfg(feature = "code-mode")]
                    ctx.insert(crate::registry::ToolTypeScriptDefs(
                        self.registry.to_typescript_defs(),
                    ));
                    let tokens_after = ctx.total_tokens(counter);
                    return LayerResult {
                        layer: self.name().into(),
                        tokens_before,
                        tokens_after,
                        duration: Duration::ZERO,
                        detail: "reused cached catalog (tools unchanged)".into(),
                    };
                }
            }
        }

        let (catalog_tokens, full_tokens) = self.registry.token_savings();
        let tool_count = self.registry.len();

        // Replace tools with just the search meta-tool
        ctx.tools = vec![self.registry.search_tool_spec()];
        ctx.catalog = Some(self.registry.catalog().to_string());

        // Pre-compute TS defs for CodeModeLayer
        #[cfg(feature = "code-mode")]
        ctx.insert(crate::registry::ToolTypeScriptDefs(
            self.registry.to_typescript_defs(),
        ));

        // Update optimization state with cached catalog
        let cached_catalog = ctx.catalog.clone();
        if let Some(state) = ctx.get_mut::<OptimizationState>() {
            state.tools_hash = Some(current_hash);
            state.cached_catalog = cached_catalog;
        }

        let tokens_after = ctx.total_tokens(counter);

        LayerResult {
            layer: self.name().into(),
            tokens_before,
            tokens_after,
            duration: Duration::ZERO,
            detail: format!("{tool_count} tools: {full_tokens} → {catalog_tokens} catalog tokens"),
        }
    }

    fn handle_tool_call(&self, tool_name: &str, args: &serde_json::Value) -> Option<String> {
        if tool_name != "tool_search" {
            return None;
        }

        let query = args.get("query").and_then(|v| v.as_str()).unwrap_or("");

        let detail = args
            .get("detail")
            .and_then(|v| v.as_str())
            .unwrap_or("full");

        // An absent query is a caller fault, not an empty result. Reporting it
        // as "no tools found" sends the caller off to rewrite a query it never
        // sent, and hides the real cause — arguments that arrived as a string,
        // or under a different key.
        if query.is_empty() {
            return Some(
                "tool_search needs a non-empty `query` string. Pass arguments as \
                 an object, for example {\"query\": \"read a file\"}."
                    .to_string(),
            );
        }

        let results = self.registry.search(query, 5);

        if results.is_empty() {
            return Some(format!(
                "No tools found matching \"{query}\". Try different keywords."
            ));
        }

        Some(self.registry.format_results(&results, detail))
    }
}
