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

    /// The judge returned no usable probes.
    ///
    /// Distinct from an empty result on purpose. Returning `Ok(vec![])` made a
    /// judge that ignored the output format indistinguishable from a context
    /// with nothing worth asking, so a caller scored perfect retention against
    /// zero questions.
    #[error(
        "no probes parsed from {lines} response line(s); the judge did not follow the TYPE|QUESTION|ANSWER format"
    )]
    NoProbesParsed { lines: usize },

    #[error("tool execution failed: {0}")]
    ToolExecution(String),

    #[error("configuration error: {0}")]
    Config(String),

    #[error("probe evaluation failed: {0}")]
    ProbeEvaluation(String),
}

pub type Result<T> = std::result::Result<T, Error>;
