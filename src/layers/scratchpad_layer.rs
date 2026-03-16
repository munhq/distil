use std::collections::BTreeMap;
use std::sync::{Arc, RwLock};
use std::time::Duration;

use crate::counter::TokenCounter;
use crate::pipeline::{Ctx, Layer, LayerResult};
use crate::types::{Role, ToolSpec};

/// Agent working memory that lives outside the context window.
///
/// The scratchpad lets agents persist notes, findings, and state across
/// conversation turns and compaction events. Unlike in-context memory
/// (which gets trimmed), scratchpad entries survive indefinitely.
///
/// The layer:
/// 1. Injects `note_write` and `note_read` tools into `ctx.tools`
/// 2. Prepends a compact summary of stored entries to the system prompt
///
/// Handle `note_write`/`note_read` calls via [`Pipeline::handle_tool_call`].
///
/// ## Usage
///
/// ```rust,ignore
/// let scratchpad = ScratchpadLayer::new();
///
/// // In your agent loop, after LLM responds:
/// if let Some(output) = pipeline.handle_tool_call("note_write", &args) {
///     // Include output in conversation
/// }
/// ```
pub struct ScratchpadLayer {
    entries: Arc<RwLock<BTreeMap<String, String>>>,
    /// Max entries before oldest are evicted.
    max_entries: usize,
    /// Max tokens per individual value.
    max_value_tokens: usize,
}

impl ScratchpadLayer {
    pub fn new() -> Self {
        Self {
            entries: Arc::new(RwLock::new(BTreeMap::new())),
            max_entries: 50,
            max_value_tokens: 500,
        }
    }

    pub fn max_entries(mut self, max: usize) -> Self {
        self.max_entries = max;
        self
    }

    pub fn max_value_tokens(mut self, max: usize) -> Self {
        self.max_value_tokens = max;
        self
    }

    /// Directly set a scratchpad entry (for pre-populating).
    pub fn set(&self, key: String, value: String) {
        let mut entries = self.entries.write().unwrap();
        entries.insert(key, value);
        self.enforce_limit(&mut entries);
    }

    /// Directly get a scratchpad entry.
    pub fn get(&self, key: &str) -> Option<String> {
        self.entries.read().unwrap().get(key).cloned()
    }

    /// List all keys.
    pub fn keys(&self) -> Vec<String> {
        self.entries.read().unwrap().keys().cloned().collect()
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.entries.read().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.read().unwrap().is_empty()
    }

    /// Generate a compact summary of all entries for injection into context.
    fn summary(&self) -> Option<String> {
        let entries = self.entries.read().unwrap();
        if entries.is_empty() {
            return None;
        }

        let mut out = String::from("## Agent Notes\n");
        for (key, value) in entries.iter() {
            let preview = if value.len() > 200 {
                format!("{}...", &value[..197])
            } else {
                value.clone()
            };
            out.push_str(&format!("- **{key}**: {preview}\n"));
        }
        Some(out)
    }

    fn enforce_limit(&self, entries: &mut BTreeMap<String, String>) {
        while entries.len() > self.max_entries {
            // Remove the first (alphabetically earliest) key
            if let Some(first_key) = entries.keys().next().cloned() {
                entries.remove(&first_key);
            }
        }
    }

    fn tool_specs() -> Vec<ToolSpec> {
        vec![
            ToolSpec {
                name: "note_write".into(),
                description: "Save a note to persistent memory. Survives context compaction. Use for tracking progress, decisions, and key findings.".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "key": {
                            "type": "string",
                            "description": "Short identifier for this note (e.g. 'plan', 'findings', 'blockers')"
                        },
                        "value": {
                            "type": "string",
                            "description": "The content to store"
                        }
                    },
                    "required": ["key", "value"]
                }),
            },
            ToolSpec {
                name: "note_read".into(),
                description: "Read a note from persistent memory by key, or list all keys if no key given.".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "key": {
                            "type": "string",
                            "description": "The key to read. Omit to list all keys."
                        }
                    }
                }),
            },
        ]
    }
}

impl Default for ScratchpadLayer {
    fn default() -> Self {
        Self::new()
    }
}

impl Layer for ScratchpadLayer {
    fn name(&self) -> &str {
        "scratchpad"
    }

    fn phase(&self) -> Option<crate::pipeline::Phase> {
        Some(crate::pipeline::Phase::Finalize)
    }

    fn apply(&self, ctx: &mut Ctx, counter: &dyn TokenCounter) -> LayerResult {
        let tokens_before = ctx.total_tokens(counter);

        // Inject scratchpad tools
        ctx.tools.extend(Self::tool_specs());

        // Inject summary into the first system message (or create one)
        if let Some(summary) = self.summary() {
            if let Some(sys_msg) = ctx.messages.iter_mut().find(|m| m.role == Role::System) {
                sys_msg.content.push_str("\n\n");
                sys_msg.content.push_str(&summary);
            } else {
                ctx.messages.insert(0, crate::types::Message::system(summary));
            }
        }

        let tokens_after = ctx.total_tokens(counter);
        let entry_count = self.len();

        LayerResult {
            layer: self.name().into(),
            tokens_before,
            tokens_after,
            duration: Duration::ZERO,
            detail: format!("{entry_count} notes injected"),
        }
    }

    fn handle_tool_call(
        &self,
        tool_name: &str,
        args: &serde_json::Value,
    ) -> Option<String> {
        match tool_name {
            "note_write" => {
                let key = args.get("key")?.as_str()?;
                let value = args.get("value")?.as_str()?;
                self.set(key.to_string(), value.to_string());
                Some(format!("Saved note \"{key}\" ({} chars)", value.len()))
            }
            "note_read" => {
                if let Some(key) = args.get("key").and_then(|v| v.as_str()) {
                    match self.get(key) {
                        Some(value) => Some(value),
                        None => Some(format!("No note found with key \"{key}\"")),
                    }
                } else {
                    let keys = self.keys();
                    if keys.is_empty() {
                        Some("No notes stored yet.".into())
                    } else {
                        Some(format!("Stored notes: {}", keys.join(", ")))
                    }
                }
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::counter::EstimateCounter;
    use crate::types::Message;

    #[test]
    fn write_and_read_entries() {
        let pad = ScratchpadLayer::new();
        pad.set("plan".into(), "Step 1: do X, Step 2: do Y".into());
        pad.set("findings".into(), "The bug is in module Z".into());

        assert_eq!(pad.len(), 2);
        assert_eq!(pad.get("plan").unwrap(), "Step 1: do X, Step 2: do Y");
        assert!(pad.get("nonexistent").is_none());
    }

    #[test]
    fn enforces_max_entries() {
        let pad = ScratchpadLayer::new().max_entries(3);
        for i in 0..10 {
            pad.set(format!("key_{i:02}"), format!("value {i}"));
        }
        assert_eq!(pad.len(), 3);
    }

    #[test]
    fn injects_tools_and_summary() {
        let counter = EstimateCounter;
        let pad = ScratchpadLayer::new();
        pad.set("plan".into(), "Fix the auth module".into());

        let mut ctx = Ctx::new(
            vec![Message::system("You are helpful.")],
            vec![],
            0,
        );

        pad.apply(&mut ctx, &counter);

        // Should have injected 2 tools
        assert_eq!(ctx.tools.len(), 2);
        let tool_names: Vec<&str> = ctx.tools.iter().map(|t| t.name.as_str()).collect();
        assert!(tool_names.contains(&"note_write"));
        assert!(tool_names.contains(&"note_read"));

        // Should have injected summary into system prompt
        assert!(ctx.messages[0].content.contains("Agent Notes"));
        assert!(ctx.messages[0].content.contains("plan"));
    }

    #[test]
    fn handle_tool_calls() {
        let pad = ScratchpadLayer::new();

        // Write
        let result = pad.handle_tool_call(
            "note_write",
            &serde_json::json!({"key": "test", "value": "hello world"}),
        );
        assert!(result.is_some());
        assert!(result.unwrap().contains("Saved"));

        // Read
        let result = pad.handle_tool_call(
            "note_read",
            &serde_json::json!({"key": "test"}),
        );
        assert_eq!(result.unwrap(), "hello world");

        // List
        let result = pad.handle_tool_call(
            "note_read",
            &serde_json::json!({}),
        );
        assert!(result.unwrap().contains("test"));
    }

    #[test]
    fn ignores_unrelated_tool_calls() {
        let pad = ScratchpadLayer::new();
        assert!(pad.handle_tool_call("shell", &serde_json::json!({})).is_none());
    }
}
