use thiserror::Error;

/// Errors that can occur during context optimization.
#[derive(Debug, Error)]
pub enum Error {
    #[error("empty context: no messages to optimize")]
    EmptyContext,

    #[error("token budget exceeded: {used} used, {budget} budget")]
    BudgetExceeded { used: usize, budget: usize },

    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("summarization failed: {0}")]
    Summarization(String),
}

pub type Result<T> = std::result::Result<T, Error>;
