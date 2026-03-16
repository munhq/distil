use crate::counter::TokenCounter;
use crate::types::{ToolSpec, ToolSummary};

/// Dynamic tool registry that replaces injecting all tool schemas upfront.
///
/// Instead of dumping 40+ full JSON schemas into every LLM request (~10K+ tokens),
/// the registry provides:
/// 1. A **compact catalog** (~500 tokens) listing tool names + one-line descriptions
/// 2. A **`tool_search` meta-tool** the LLM calls to load full schemas on demand
///
/// Typical savings: 85-95% reduction in tool definition tokens.
pub struct ToolRegistry {
    tools: Vec<ToolSpec>,
    summaries: Vec<ToolSummary>,
    catalog_text: String,
    catalog_tokens: usize,
    full_tokens: usize,
}

impl ToolRegistry {
    /// Build a registry from a list of tool specifications.
    pub fn new(tools: Vec<ToolSpec>, counter: &dyn TokenCounter) -> Self {
        let summaries: Vec<ToolSummary> = tools
            .iter()
            .map(|t| ToolSummary {
                name: t.name.clone(),
                brief: first_sentence(&t.description),
            })
            .collect();

        let catalog_text = Self::build_catalog(&summaries);
        let catalog_tokens = counter.count(&catalog_text);

        let full_tokens: usize = tools.iter().map(|t| counter.count(&t.to_prompt_text())).sum();

        Self {
            tools,
            summaries,
            catalog_text,
            catalog_tokens,
            full_tokens,
        }
    }

    /// The compact catalog string to include in the system prompt.
    ///
    /// Contains tool names and one-line descriptions. Typically ~500 tokens
    /// regardless of how many tools you have.
    pub fn catalog(&self) -> &str {
        &self.catalog_text
    }

    /// Token count of the catalog vs. the full tool definitions.
    pub fn token_savings(&self) -> (usize, usize) {
        (self.catalog_tokens, self.full_tokens)
    }

    /// The `tool_search` meta-tool spec to add to your tool list.
    ///
    /// When the LLM calls this tool, pass the arguments to [`search`] and
    /// include the returned schemas in the conversation.
    pub fn search_tool_spec(&self) -> ToolSpec {
        ToolSpec {
            name: "tool_search".into(),
            description: "Search for available tools by keyword. Returns full tool schemas for matching tools. Use this before calling a tool you haven't used yet.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Keywords to search for (e.g. 'file operations', 'git', 'memory')"
                    }
                },
                "required": ["query"]
            }),
        }
    }

    /// Search tools by keyword query. Returns full specs for matching tools.
    ///
    /// Matching is case-insensitive against tool names and descriptions.
    /// Returns up to `max_results` matches, ranked by relevance.
    pub fn search(&self, query: &str, max_results: usize) -> Vec<&ToolSpec> {
        let query_lower = query.to_lowercase();
        let query_terms: Vec<&str> = query_lower.split_whitespace().collect();

        if query_terms.is_empty() {
            return vec![];
        }

        let mut scored: Vec<(usize, &ToolSpec)> = self
            .tools
            .iter()
            .filter_map(|tool| {
                let score = relevance_score(&query_terms, &tool.name, &tool.description);
                if score > 0 {
                    Some((score, tool))
                } else {
                    None
                }
            })
            .collect();

        scored.sort_by(|a, b| b.0.cmp(&a.0));
        scored.into_iter().take(max_results).map(|(_, t)| t).collect()
    }

    /// Look up a tool by exact name.
    pub fn get(&self, name: &str) -> Option<&ToolSpec> {
        self.tools.iter().find(|t| t.name == name)
    }

    /// All tool summaries.
    pub fn summaries(&self) -> &[ToolSummary] {
        &self.summaries
    }

    /// Access the registered tool specs.
    pub fn tools(&self) -> &[ToolSpec] {
        &self.tools
    }

    /// Number of registered tools.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Generate TypeScript-style function signatures for code mode.
    pub fn to_typescript_defs(&self) -> String {
        let mut out = String::from("// Available tool functions (call with JSON string arg, returns string):\n");
        for tool in &self.tools {
            out.push_str(&Self::ts_signature(tool));
            out.push('\n');
        }
        out
    }

    fn ts_signature(tool: &ToolSpec) -> String {
        let params = Self::json_schema_to_ts_params(&tool.parameters);
        format!("function {}(args: {{{}}}): string;", tool.name, params)
    }

    fn json_schema_to_ts_params(schema: &serde_json::Value) -> String {
        let props = match schema.get("properties").and_then(|p| p.as_object()) {
            Some(p) => p,
            None => return String::new(),
        };
        let required: Vec<&str> = schema
            .get("required")
            .and_then(|r| r.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect())
            .unwrap_or_default();

        props
            .iter()
            .map(|(name, prop)| {
                let ts_type = Self::json_type_to_ts(prop);
                let optional = if required.contains(&name.as_str()) { "" } else { "?" };
                format!("{name}{optional}: {ts_type}")
            })
            .collect::<Vec<_>>()
            .join(", ")
    }

    fn json_type_to_ts(prop: &serde_json::Value) -> &'static str {
        match prop.get("type").and_then(|t| t.as_str()) {
            Some("string") => "string",
            Some("integer") | Some("number") => "number",
            Some("boolean") => "boolean",
            Some("array") => "unknown[]",
            Some("object") => "Record<string, unknown>",
            _ => "unknown",
        }
    }

    fn build_catalog(summaries: &[ToolSummary]) -> String {
        let mut out = String::from("## Available Tools\n\nUse `tool_search` to get the full schema for any tool before calling it.\n\n");
        for s in summaries {
            out.push_str(&format!("{s}\n"));
        }
        out
    }
}

/// TypeScript type definitions for tools, for cross-layer use.
#[derive(Debug, Clone)]
pub struct ToolTypeScriptDefs(pub String);

/// Score how relevant a tool is to the query terms.
/// Higher = more relevant.
fn relevance_score(query_terms: &[&str], name: &str, description: &str) -> usize {
    let name_lower = name.to_lowercase();
    let desc_lower = description.to_lowercase();
    let mut score = 0;

    for term in query_terms {
        // Exact name match is highest signal
        if name_lower == *term {
            score += 10;
        } else if name_lower.contains(term) {
            score += 5;
        }

        // Description match
        if desc_lower.contains(term) {
            score += 2;
        }
    }

    score
}

/// Extract the first sentence from a description.
fn first_sentence(text: &str) -> String {
    // Take up to the first period followed by a space, or the first newline
    let trimmed = text.trim();
    if let Some(pos) = trimmed.find(". ") {
        trimmed[..=pos].to_string()
    } else if let Some(pos) = trimmed.find('\n') {
        trimmed[..pos].trim().to_string()
    } else if trimmed.len() > 120 {
        format!("{}...", &trimmed[..117])
    } else {
        trimmed.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::counter::EstimateCounter;

    fn sample_tools() -> Vec<ToolSpec> {
        vec![
            ToolSpec {
                name: "shell".into(),
                description: "Execute a shell command. Returns stdout, stderr, and exit code.".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "command": { "type": "string", "description": "The command to execute" },
                        "timeout": { "type": "integer", "description": "Timeout in seconds" }
                    },
                    "required": ["command"]
                }),
            },
            ToolSpec {
                name: "file_read".into(),
                description: "Read a file from disk. Supports text files and returns content as a string. Use line_start/line_end for partial reads of large files.".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string" },
                        "line_start": { "type": "integer" },
                        "line_end": { "type": "integer" }
                    },
                    "required": ["path"]
                }),
            },
            ToolSpec {
                name: "file_write".into(),
                description: "Write content to a file. Creates the file if it doesn't exist, overwrites if it does.".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string" },
                        "content": { "type": "string" }
                    },
                    "required": ["path", "content"]
                }),
            },
            ToolSpec {
                name: "git_status".into(),
                description: "Get the current git status. Shows staged, unstaged, and untracked files.".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {},
                }),
            },
            ToolSpec {
                name: "memory_store".into(),
                description: "Store a key-value pair in the agent's persistent memory. Used for remembering facts across conversations.".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "key": { "type": "string" },
                        "value": { "type": "string" }
                    },
                    "required": ["key", "value"]
                }),
            },
            ToolSpec {
                name: "memory_recall".into(),
                description: "Recall a value from persistent memory by key or search query.".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "query": { "type": "string" }
                    },
                    "required": ["query"]
                }),
            },
            ToolSpec {
                name: "http_request".into(),
                description: "Make an HTTP request. Supports GET, POST, PUT, DELETE with headers and body.".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "method": { "type": "string", "enum": ["GET", "POST", "PUT", "DELETE"] },
                        "url": { "type": "string" },
                        "headers": { "type": "object" },
                        "body": { "type": "string" }
                    },
                    "required": ["method", "url"]
                }),
            },
        ]
    }

    #[test]
    fn catalog_is_much_smaller_than_full_schemas() {
        let counter = EstimateCounter;
        let registry = ToolRegistry::new(sample_tools(), &counter);

        let (catalog_tokens, full_tokens) = registry.token_savings();

        // Catalog should be significantly smaller
        assert!(
            catalog_tokens < full_tokens / 2,
            "catalog ({catalog_tokens}) should be less than half of full schemas ({full_tokens})"
        );
    }

    #[test]
    fn search_finds_relevant_tools() {
        let counter = EstimateCounter;
        let registry = ToolRegistry::new(sample_tools(), &counter);

        let results = registry.search("file", 10);
        let names: Vec<&str> = results.iter().map(|t| t.name.as_str()).collect();

        assert!(names.contains(&"file_read"));
        assert!(names.contains(&"file_write"));
        // file_read/file_write should rank higher than git_status
        // (git_status mentions "files" in description but "file" is in the name of file_*)
        assert_eq!(names[0], "file_read");
    }

    #[test]
    fn search_ranks_exact_name_higher() {
        let counter = EstimateCounter;
        let registry = ToolRegistry::new(sample_tools(), &counter);

        let results = registry.search("shell", 10);
        assert_eq!(results[0].name, "shell");
    }

    #[test]
    fn search_by_description_keyword() {
        let counter = EstimateCounter;
        let registry = ToolRegistry::new(sample_tools(), &counter);

        let results = registry.search("persistent memory", 10);
        let names: Vec<&str> = results.iter().map(|t| t.name.as_str()).collect();

        assert!(names.contains(&"memory_store"));
        assert!(names.contains(&"memory_recall"));
    }

    #[test]
    fn search_empty_query_returns_nothing() {
        let counter = EstimateCounter;
        let registry = ToolRegistry::new(sample_tools(), &counter);

        assert!(registry.search("", 10).is_empty());
    }

    #[test]
    fn get_by_name() {
        let counter = EstimateCounter;
        let registry = ToolRegistry::new(sample_tools(), &counter);

        assert!(registry.get("shell").is_some());
        assert!(registry.get("nonexistent").is_none());
    }

    #[test]
    fn typescript_defs_basic() {
        let tools = vec![
            ToolSpec {
                name: "shell".into(),
                description: "Run a command".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "timeout": {"type": "integer"}
                    },
                    "required": ["command"]
                }),
            },
        ];
        let counter = EstimateCounter;
        let registry = ToolRegistry::new(tools, &counter);
        let ts = registry.to_typescript_defs();
        assert!(ts.contains("function shell"), "should have function: {ts}");
        assert!(ts.contains("command: string"), "should have required param: {ts}");
        assert!(ts.contains("timeout?: number"), "should have optional param: {ts}");
    }

    #[test]
    fn typescript_defs_multiple_tools() {
        let counter = EstimateCounter;
        let registry = ToolRegistry::new(sample_tools(), &counter);
        let ts = registry.to_typescript_defs();
        assert!(ts.contains("function shell"), "should have shell: {ts}");
        assert!(ts.contains("function file_read"), "should have file_read: {ts}");
        assert!(ts.contains("function git_status"), "should have git_status: {ts}");
    }

    #[test]
    fn search_tool_spec_is_valid() {
        let counter = EstimateCounter;
        let registry = ToolRegistry::new(sample_tools(), &counter);

        let spec = registry.search_tool_spec();
        assert_eq!(spec.name, "tool_search");
        assert!(spec.parameters["properties"]["query"].is_object());
    }
}
