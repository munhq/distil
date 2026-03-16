use serde::{Deserialize, Serialize};

/// Role in a conversation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

/// A message in the conversation.
///
/// Framework-agnostic: just role + content string. Distil doesn't impose
/// a specific message format — it works with whatever your agent uses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: content.into(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into(),
        }
    }

    pub fn tool(content: impl Into<String>) -> Self {
        Self {
            role: Role::Tool,
            content: content.into(),
        }
    }

    /// Return the role as a static string slice.
    pub fn role_str(&self) -> &'static str {
        match self.role {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
        }
    }
}

/// A tool's full specification (name + description + JSON Schema parameters).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

impl ToolSpec {
    /// Render this tool as the text that would be injected into a system prompt.
    pub fn to_prompt_text(&self) -> String {
        format!(
            "**{}**: {}\nParameters: {}",
            self.name,
            self.description,
            serde_json::to_string(&self.parameters).unwrap_or_default()
        )
    }
}

/// A compact one-line tool summary for the catalog.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSummary {
    pub name: String,
    pub brief: String,
}

impl std::fmt::Display for ToolSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "- `{}` — {}", self.name, self.brief)
    }
}

/// Token usage broken down by component.
#[derive(Debug, Clone, Default, Serialize)]
pub struct Breakdown {
    pub system: usize,
    pub tools: usize,
    pub history: usize,
    pub tool_results: usize,
    pub total: usize,
}

/// Tokens saved by each optimization layer.
#[derive(Debug, Clone, Default, Serialize)]
pub struct Savings {
    pub registry: usize,
    pub masking: usize,
    pub total: usize,
    pub percentage: f64,
}

/// Full statistics from an optimization pass.
#[derive(Debug, Clone, Default, Serialize)]
pub struct Stats {
    pub before: Breakdown,
    pub after: Breakdown,
    pub savings: Savings,
}

impl std::fmt::Display for Stats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} → {} tokens (saved {} / {:.1}%)",
            self.before.total, self.after.total, self.savings.total, self.savings.percentage
        )
    }
}
