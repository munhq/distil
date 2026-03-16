//! Probe-based evaluation of information retention after context compression.
//!
//! Quantifies whether compression degrades the LLM's ability to answer questions
//! about the original context. Uses the [`Summarizer`] trait to call an LLM for
//! both probe generation and evaluation.
//!
//! NOT a pipeline layer -- runs as a post-pipeline check comparing original
//! and compressed context.
//!
//! ```rust,ignore
//! let evaluator = ProbeEvaluator::new(my_llm_client);
//! let original = ctx.messages.clone();
//! pipeline.optimize(&mut ctx);
//! let report = evaluator.evaluate(&original, &ctx.messages)?;
//! if report.success_rate < 0.8 {
//!     // compression is too aggressive
//! }
//! ```

use std::collections::HashMap;

use crate::summarizer::Summarizer;
use crate::types::Message;

/// Type of probe question.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProbeType {
    /// Can the LLM recall specific facts? ("What was the exit code?")
    Recall,
    /// Does the LLM know what artifacts exist? ("What files were created?")
    Artifact,
    /// Can the LLM continue multi-turn work? ("What should happen next?")
    Continuation,
    /// Is reasoning preserved? ("Why was PostgreSQL chosen?")
    Decision,
}

impl std::fmt::Display for ProbeType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProbeType::Recall => write!(f, "recall"),
            ProbeType::Artifact => write!(f, "artifact"),
            ProbeType::Continuation => write!(f, "continuation"),
            ProbeType::Decision => write!(f, "decision"),
        }
    }
}

/// A probe question with expected answer.
#[derive(Debug, Clone)]
pub struct Probe {
    pub probe_type: ProbeType,
    pub question: String,
    pub expected: String,
}

/// Result of evaluating one probe.
#[derive(Debug, Clone)]
pub struct ProbeResult {
    pub probe: Probe,
    pub passed: bool,
    pub actual_answer: String,
}

/// Aggregate results from all probes.
#[derive(Debug, Clone)]
pub struct ProbeReport {
    pub results: Vec<ProbeResult>,
    pub success_rate: f64,
    pub by_type: HashMap<ProbeType, f64>,
}

impl ProbeReport {
    fn compute(results: Vec<ProbeResult>) -> Self {
        let total = results.len();
        let passed = results.iter().filter(|r| r.passed).count();
        let success_rate = if total > 0 {
            passed as f64 / total as f64
        } else {
            1.0
        };

        let mut type_counts: HashMap<ProbeType, (usize, usize)> = HashMap::new();
        for r in &results {
            let entry = type_counts.entry(r.probe.probe_type).or_insert((0, 0));
            entry.0 += 1;
            if r.passed {
                entry.1 += 1;
            }
        }

        let by_type = type_counts
            .into_iter()
            .map(|(t, (total, passed))| (t, passed as f64 / total as f64))
            .collect();

        ProbeReport {
            results,
            success_rate,
            by_type,
        }
    }
}

impl std::fmt::Display for ProbeReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "Probe Report: {:.1}% success ({}/{})",
            self.success_rate * 100.0,
            self.results.iter().filter(|r| r.passed).count(),
            self.results.len()
        )?;
        for (probe_type, rate) in &self.by_type {
            writeln!(f, "  {}: {:.1}%", probe_type, rate * 100.0)?;
        }
        Ok(())
    }
}

/// Evaluates information retention by generating and checking probe questions.
///
/// NOT a pipeline layer -- runs as a post-pipeline check comparing original
/// and compressed context.
///
/// ```rust,ignore
/// let evaluator = ProbeEvaluator::new(my_llm_client);
/// let original = ctx.messages.clone();
/// pipeline.optimize(&mut ctx);
/// let report = evaluator.evaluate(&original, &ctx.messages)?;
/// if report.success_rate < 0.8 {
///     // compression is too aggressive
/// }
/// ```
pub struct ProbeEvaluator<S: Summarizer> {
    evaluator: S,
    num_probes: usize,
    threshold: f64,
}

impl<S: Summarizer> ProbeEvaluator<S> {
    pub fn new(evaluator: S) -> Self {
        Self {
            evaluator,
            num_probes: 5,
            threshold: 0.8,
        }
    }

    pub fn num_probes(mut self, n: usize) -> Self {
        self.num_probes = n;
        self
    }

    pub fn threshold(mut self, t: f64) -> Self {
        self.threshold = t;
        self
    }

    pub fn threshold_value(&self) -> f64 {
        self.threshold
    }

    /// Generate probe questions from the original (uncompressed) context.
    pub fn generate_probes(&self, original: &[Message]) -> crate::error::Result<Vec<Probe>> {
        let context = original
            .iter()
            .map(|m| format!("[{}]: {}", m.role_str(), m.content))
            .collect::<Vec<_>>()
            .join("\n");

        let prompt = format!(
            "Given this conversation, generate exactly {} factual questions with short answers.\n\
             Each question should test whether key information is preserved.\n\
             Format each as: TYPE|QUESTION|ANSWER\n\
             Types: recall, artifact, continuation, decision\n\n\
             Conversation:\n{}\n\n\
             Generate {} questions:",
            self.num_probes, context, self.num_probes
        );

        let response = self.evaluator.summarize(&prompt, 500)?;
        Self::parse_probes(&response)
    }

    /// Check if the compressed context can answer the probe questions.
    pub fn evaluate_probes(
        &self,
        compressed: &[Message],
        probes: &[Probe],
    ) -> crate::error::Result<ProbeReport> {
        let context = compressed
            .iter()
            .map(|m| format!("[{}]: {}", m.role_str(), m.content))
            .collect::<Vec<_>>()
            .join("\n");

        let mut results = Vec::new();
        for probe in probes {
            let prompt = format!(
                "Based ONLY on this conversation context, answer the question.\n\
                 If the information is not available, say \"UNKNOWN\".\n\n\
                 Context:\n{}\n\n\
                 Question: {}\n\
                 Answer:",
                context, probe.question
            );

            let answer = self.evaluator.summarize(&prompt, 100)?;
            let passed = Self::check_answer(&answer, &probe.expected);
            results.push(ProbeResult {
                probe: probe.clone(),
                passed,
                actual_answer: answer,
            });
        }

        Ok(ProbeReport::compute(results))
    }

    /// Generate probes from original context and evaluate against compressed context.
    pub fn evaluate(
        &self,
        original: &[Message],
        compressed: &[Message],
    ) -> crate::error::Result<ProbeReport> {
        let probes = self.generate_probes(original)?;
        self.evaluate_probes(compressed, &probes)
    }

    fn parse_probes(response: &str) -> crate::error::Result<Vec<Probe>> {
        let mut probes = Vec::new();
        for line in response.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let parts: Vec<&str> = line.splitn(3, '|').collect();
            if parts.len() != 3 {
                continue;
            }

            let probe_type = match parts[0].trim().to_lowercase().as_str() {
                "recall" => ProbeType::Recall,
                "artifact" => ProbeType::Artifact,
                "continuation" => ProbeType::Continuation,
                "decision" => ProbeType::Decision,
                _ => continue,
            };

            probes.push(Probe {
                probe_type,
                question: parts[1].trim().to_string(),
                expected: parts[2].trim().to_string(),
            });
        }
        Ok(probes)
    }

    fn check_answer(actual: &str, expected: &str) -> bool {
        let actual_lower = actual.to_lowercase();
        let expected_lower = expected.to_lowercase();

        // Exact match
        if actual_lower.contains(&expected_lower) {
            return true;
        }

        // Check if key words from expected appear in actual
        let expected_words: Vec<&str> = expected_lower
            .split_whitespace()
            .filter(|w| w.len() > 3) // skip short words
            .collect();
        if expected_words.is_empty() {
            return actual_lower.contains(&expected_lower);
        }
        let matched = expected_words
            .iter()
            .filter(|w| actual_lower.contains(**w))
            .count();
        matched as f64 / expected_words.len() as f64 > 0.5
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockEvaluator {
        generate_response: String,
        evaluate_response: String,
    }

    impl crate::Summarizer for MockEvaluator {
        fn summarize(&self, content: &str, _max_tokens: usize) -> crate::error::Result<String> {
            if content.contains("generate") || content.contains("Generate") {
                Ok(self.generate_response.clone())
            } else {
                Ok(self.evaluate_response.clone())
            }
        }
    }

    #[test]
    fn parses_probe_response() {
        let response = "recall|What was the exit code?|0\nartifact|What file was created?|src/auth.rs\ndecision|Why PostgreSQL?|Better for relational data";
        let probes = ProbeEvaluator::<MockEvaluator>::parse_probes(response).unwrap();
        assert_eq!(probes.len(), 3);
        assert_eq!(probes[0].probe_type, ProbeType::Recall);
        assert_eq!(probes[1].probe_type, ProbeType::Artifact);
        assert_eq!(probes[2].probe_type, ProbeType::Decision);
    }

    #[test]
    fn check_answer_exact_match() {
        assert!(ProbeEvaluator::<MockEvaluator>::check_answer(
            "exit code was 0",
            "0"
        ));
        assert!(ProbeEvaluator::<MockEvaluator>::check_answer(
            "The file src/auth.rs was created",
            "src/auth.rs"
        ));
    }

    #[test]
    fn check_answer_keyword_match() {
        assert!(ProbeEvaluator::<MockEvaluator>::check_answer(
            "PostgreSQL was chosen because it handles relational data better",
            "Better for relational data"
        ));
    }

    #[test]
    fn check_answer_unknown_fails() {
        assert!(!ProbeEvaluator::<MockEvaluator>::check_answer(
            "UNKNOWN",
            "src/auth.rs"
        ));
    }

    #[test]
    fn probe_report_computes_rates() {
        let results = vec![
            ProbeResult {
                probe: Probe {
                    probe_type: ProbeType::Recall,
                    question: "q1".into(),
                    expected: "a1".into(),
                },
                passed: true,
                actual_answer: "a1".into(),
            },
            ProbeResult {
                probe: Probe {
                    probe_type: ProbeType::Recall,
                    question: "q2".into(),
                    expected: "a2".into(),
                },
                passed: false,
                actual_answer: "wrong".into(),
            },
            ProbeResult {
                probe: Probe {
                    probe_type: ProbeType::Artifact,
                    question: "q3".into(),
                    expected: "a3".into(),
                },
                passed: true,
                actual_answer: "a3".into(),
            },
        ];
        let report = ProbeReport::compute(results);
        assert!((report.success_rate - 0.6667).abs() < 0.01);
        assert!((report.by_type[&ProbeType::Recall] - 0.5).abs() < 0.01);
        assert!((report.by_type[&ProbeType::Artifact] - 1.0).abs() < 0.01);
    }

    #[test]
    fn evaluate_with_mock() {
        let evaluator = ProbeEvaluator::new(MockEvaluator {
            generate_response: "recall|What was the exit code?|0\nartifact|What file was created?|src/auth.rs".into(),
            evaluate_response: "The exit code was 0 and src/auth.rs was created".into(),
        })
        .num_probes(2);

        let original = vec![
            Message::user("Build auth"),
            Message::assistant("Created src/auth.rs, exit code 0"),
        ];
        let compressed = vec![
            Message::user("Build auth"),
            Message::assistant("[summary: auth built]"),
        ];

        let report = evaluator.evaluate(&original, &compressed).unwrap();
        assert_eq!(report.results.len(), 2);
        // Both should pass because the evaluate_response contains the expected answers
        assert!(report.success_rate > 0.5);
    }
}
