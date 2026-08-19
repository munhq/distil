use std::time::Duration;

use crate::counter::TokenCounter;
use crate::pipeline::{Ctx, Layer, LayerResult};
use crate::types::Role;

/// Reorders system prompt content to maximize prompt cache hit rates.
///
/// LLM providers (Anthropic, OpenAI) cache the prefix of the prompt. If your
/// system prompt starts with the same content across requests, subsequent
/// requests get a cache hit and the cached portion is free (or heavily discounted).
///
/// The trick: put **stable content first** (instructions, tool catalog) and
/// **dynamic content last** (scratchpad summary, session-specific context).
/// This maximizes the stable prefix length = more cache hits.
///
/// This layer restructures the system message(s) to follow this ordering:
///
/// 1. **Static instructions** (your base system prompt — rarely changes)
/// 2. **Tool catalog** (changes only when tools are added/removed)
/// 3. **Scratchpad/notes** (changes each turn)
/// 4. **Session context** (conversation-specific, always changing)
///
/// ## Provider-Specific Notes
///
/// - **Anthropic**: Caches the longest matching prefix. Place cache breakpoints
///   after stable sections. Minimum cacheable prefix: 1024 tokens.
/// - **OpenAI**: Similar prefix caching. No explicit breakpoints needed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheProvider {
    Anthropic,
    OpenAi,
    /// No reordering, just merge multiple system messages.
    Generic,
}

pub struct CacheAlignLayer {
    provider: CacheProvider,
}

impl CacheAlignLayer {
    pub fn new(provider: CacheProvider) -> Self {
        Self { provider }
    }

    pub fn anthropic() -> Self {
        Self::new(CacheProvider::Anthropic)
    }

    pub fn openai() -> Self {
        Self::new(CacheProvider::OpenAi)
    }

    pub fn generic() -> Self {
        Self::new(CacheProvider::Generic)
    }
}

impl Layer for CacheAlignLayer {
    fn name(&self) -> &str {
        "cache_align"
    }

    fn phase(&self) -> Option<crate::pipeline::Phase> {
        Some(crate::pipeline::Phase::Finalize)
    }

    fn apply(&self, ctx: &mut Ctx, counter: &dyn TokenCounter) -> LayerResult {
        let tokens_before = ctx.total_tokens(counter);

        // Collect all system messages
        let system_parts: Vec<String> = ctx
            .messages
            .iter()
            .filter(|m| m.role == Role::System)
            .map(|m| m.content.clone())
            .collect();

        if system_parts.is_empty() {
            return LayerResult {
                layer: self.name().into(),
                tokens_before,
                tokens_after: tokens_before,
                duration: Duration::ZERO,
                detail: "no system messages to align".into(),
            };
        }

        // Remove all system messages
        ctx.messages.retain(|m| m.role != Role::System);

        // Build the unified system message with cache-friendly ordering:
        // 1. Static instructions (the original system prompt content)
        // 2. Tool catalog (if present)
        // 3. Dynamic content (scratchpad notes, etc.)
        let mut unified = String::new();

        // Separate static from dynamic content within system messages.
        // Heuristic: "## Agent Notes" and similar headers mark dynamic sections.
        let mut static_parts = Vec::new();
        let mut dynamic_parts = Vec::new();

        for part in &system_parts {
            if let Some(split_pos) = find_dynamic_boundary(part) {
                static_parts.push(part[..split_pos].trim_end().to_string());
                dynamic_parts.push(part[split_pos..].to_string());
            } else {
                static_parts.push(part.clone());
            }
        }

        // Static first
        for part in &static_parts {
            if !unified.is_empty() {
                unified.push_str("\n\n");
            }
            unified.push_str(part);
        }

        // Then tool catalog
        if let Some(catalog) = ctx.catalog.take() {
            unified.push_str("\n\n");
            unified.push_str(&catalog);

            // For Anthropic: add a cache breakpoint marker after the catalog
            if self.provider == CacheProvider::Anthropic {
                unified.push_str("\n<!-- cache_breakpoint -->");
            }
        }

        // Then dynamic content
        for part in &dynamic_parts {
            unified.push_str("\n\n");
            unified.push_str(part);
        }

        // Insert the unified system message at position 0
        ctx.messages
            .insert(0, crate::types::Message::system(unified));

        let tokens_after = ctx.total_tokens(counter);
        let merged_count = system_parts.len();

        LayerResult {
            layer: self.name().into(),
            tokens_before,
            tokens_after,
            duration: Duration::ZERO,
            detail: format!(
                "merged {merged_count} system messages, {:?} ordering",
                self.provider
            ),
        }
    }
}

/// Find the boundary between static and dynamic content in a system message.
/// Returns the byte offset where dynamic content begins, or None if all static.
fn find_dynamic_boundary(content: &str) -> Option<usize> {
    let dynamic_markers = [
        "\n## Agent Notes",
        "\n## Session Context",
        "\n## Current State",
        "\n## Working Memory",
        "\n<!-- dynamic -->",
    ];

    dynamic_markers
        .iter()
        .filter_map(|marker| content.find(marker))
        .min()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::counter::EstimateCounter;
    use crate::types::Message;

    #[test]
    fn merges_multiple_system_messages() {
        let counter = EstimateCounter;
        let layer = CacheAlignLayer::generic();

        let mut ctx = Ctx::new(
            vec![
                Message::system("You are a helpful assistant."),
                Message::system("Always be concise."),
                Message::user("Hello"),
            ],
            vec![],
            0,
        );

        layer.apply(&mut ctx, &counter);

        // Should have merged into one system message
        let system_msgs: Vec<_> = ctx
            .messages
            .iter()
            .filter(|m| m.role == Role::System)
            .collect();
        assert_eq!(system_msgs.len(), 1);
        assert!(system_msgs[0].content.contains("helpful assistant"));
        assert!(system_msgs[0].content.contains("concise"));
    }

    #[test]
    fn places_catalog_after_static_content() {
        let counter = EstimateCounter;
        let layer = CacheAlignLayer::generic();

        let mut ctx = Ctx::new(
            vec![
                Message::system("You are helpful.\n\n## Agent Notes\n- plan: fix bugs"),
                Message::user("Hello"),
            ],
            vec![],
            0,
        );
        ctx.catalog = Some("## Available Tools\n- shell\n- file_read".into());

        layer.apply(&mut ctx, &counter);

        let sys = &ctx.messages[0].content;
        let catalog_pos = sys.find("Available Tools").unwrap();
        let notes_pos = sys.find("Agent Notes").unwrap();
        let instructions_pos = sys.find("You are helpful").unwrap();

        // Order should be: instructions < catalog < notes
        assert!(
            instructions_pos < catalog_pos,
            "instructions should come before catalog"
        );
        assert!(
            catalog_pos < notes_pos,
            "catalog ({catalog_pos}) should come before notes ({notes_pos})"
        );
    }

    #[test]
    fn anthropic_adds_cache_breakpoint() {
        let counter = EstimateCounter;
        let layer = CacheAlignLayer::anthropic();

        let mut ctx = Ctx::new(vec![Message::system("Be helpful.")], vec![], 0);
        ctx.catalog = Some("## Tools\n- shell".into());

        layer.apply(&mut ctx, &counter);

        assert!(ctx.messages[0].content.contains("cache_breakpoint"));
    }

    #[test]
    fn catalog_consumed_after_alignment() {
        let counter = EstimateCounter;
        let layer = CacheAlignLayer::generic();

        let mut ctx = Ctx::new(vec![Message::system("Be helpful.")], vec![], 0);
        ctx.catalog = Some("## Tools\n- shell".into());

        layer.apply(&mut ctx, &counter);

        // Catalog should be consumed (moved into system message)
        assert!(ctx.catalog.is_none());
    }
}
