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

use crate::summarizer::Completer;
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
pub struct ProbeEvaluator<C: Completer> {
    evaluator: C,
    num_probes: usize,
    threshold: f64,
    max_context_chars: usize,
}

impl<C: Completer> ProbeEvaluator<C> {
    pub fn new(evaluator: C) -> Self {
        Self {
            evaluator,
            num_probes: 5,
            threshold: 0.8,
            // ~6k tokens of context. Small enough for a modest local judge,
            // large enough to carry a session's decisions and its last actions.
            max_context_chars: 24_000,
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

    /// Maximum characters of context sent to the judge in one prompt.
    ///
    /// A real session runs to hundreds of thousands of characters, and a judge
    /// handed all of it at once returns nothing usable — that failure looked
    /// like a format problem until the context was cut down and the same model
    /// complied immediately.
    pub fn max_context_chars(mut self, n: usize) -> Self {
        self.max_context_chars = n;
        self
    }

    /// Render messages as text, keeping the HEAD and TAIL when too long.
    ///
    /// The middle is dropped rather than the end. Probes must cover what the
    /// session decided and what it did last; truncating to the first N
    /// characters would generate questions only about the opening, which the
    /// compressed context nearly always still contains — the measurement would
    /// then flatter every compressor.
    fn render(&self, messages: &[Message]) -> String {
        let full = messages
            .iter()
            .map(|m| format!("[{}]: {}", m.role_str(), m.content))
            .collect::<Vec<_>>()
            .join("\n");
        if full.len() <= self.max_context_chars {
            return full;
        }
        let half = self.max_context_chars / 2;
        // Split on char boundaries, never bytes, or a multi-byte character
        // straddling the cut panics.
        let head_end = full
            .char_indices()
            .map(|(i, _)| i)
            .take_while(|&i| i <= half)
            .last()
            .unwrap_or(0);
        let tail_start = full
            .char_indices()
            .map(|(i, _)| i)
            .find(|&i| i >= full.len().saturating_sub(half))
            .unwrap_or(full.len());
        format!(
            "{}\n\n[... {} characters of the middle omitted ...]\n\n{}",
            &full[..head_end],
            tail_start - head_end,
            &full[tail_start..]
        )
    }

    /// Generate probe questions from the original (uncompressed) context.
    pub fn generate_probes(&self, original: &[Message]) -> crate::error::Result<Vec<Probe>> {
        let context = self.render(original);

        // The example line and the prohibitions are load-bearing. Without them
        // models answer with a numbered list and drop the TYPE field, which
        // yields zero parseable probes — see `parse_probes` for the other half
        // of this defence.
        let prompt = format!(
            "Given this conversation, write exactly {} factual questions with short answers.\n\
             Each question must test whether key information survived compression.\n\n\
             Output one question per line, in exactly this form:\n\
             TYPE|QUESTION|ANSWER\n\n\
             TYPE is one of: recall, artifact, continuation, decision\n\n\
             Example of a correct line:\n\
             recall|What was the exit code of the build?|1\n\n\
             Rules:\n\
             - Start every line with the TYPE word. Never omit it.\n\
             - Do not number the lines.\n\
             - Do not add bullets, headers, or any other text.\n\
             - Use exactly two | characters per line.\n\n\
             Conversation:\n{}\n\n\
             Write the {} lines now:",
            self.num_probes, context, self.num_probes
        );

        let response = self.evaluator.complete(&prompt, 500)?;
        let probes = Self::parse_probes(&response)?;
        if probes.is_empty() {
            // A caller cannot act on an empty vector: it means both "nothing to
            // ask" and "the judge ignored the format", and only one of those is
            // a valid measurement.
            return Err(crate::error::Error::NoProbesParsed {
                lines: response.lines().filter(|l| !l.trim().is_empty()).count(),
            });
        }
        Ok(probes)
    }

    /// Check if the compressed context can answer the probe questions.
    pub fn evaluate_probes(
        &self,
        compressed: &[Message],
        probes: &[Probe],
    ) -> crate::error::Result<ProbeReport> {
        let context = self.render(compressed);

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

            let answer = self.evaluator.complete(&prompt, 100)?;
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

    /// Parse `TYPE|QUESTION|ANSWER` lines out of a model response.
    ///
    /// Tolerant on purpose. Models routinely wrap the requested format in a
    /// numbered list and pad it with pipes — `1. |What failed?|E0308|` — and a
    /// strict `splitn(3, '|')` reads the leading `1. ` as the type, matches no
    /// variant, and drops every probe. The failure is silent, so the caller
    /// measures nothing and believes it measured perfect retention.
    ///
    /// So leading list markers and surrounding pipes are stripped before the
    /// type is matched. Lines that still carry no recognised type are counted
    /// and returned to the caller rather than discarded quietly.
    fn parse_probes(response: &str) -> crate::error::Result<Vec<Probe>> {
        let mut probes = Vec::new();
        for line in response.lines() {
            let mut line = line.trim();

            // Strip an ordered- or bulleted-list marker: "1. ", "2) ", "- ", "* ".
            if let Some(rest) = line.strip_prefix(['-', '*']) {
                line = rest.trim_start();
            } else {
                let digits = line.chars().take_while(|c| c.is_ascii_digit()).count();
                if digits > 0 && digits < line.len() {
                    let after = &line[digits..];
                    if let Some(rest) = after.strip_prefix('.').or_else(|| after.strip_prefix(')'))
                    {
                        line = rest.trim_start();
                    }
                }
            }
            // Strip pipes used as table-style delimiters around the whole row.
            line = line.trim_matches('|').trim();
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

    impl Completer for MockEvaluator {
        fn complete(&self, content: &str, _max_tokens: usize) -> crate::error::Result<String> {
            // Branch on the format specification, which only the generation
            // prompt carries. An earlier version sniffed for the word
            // "generate" and silently mis-routed the moment that prompt was
            // reworded — the test then failed for a reason unrelated to the
            // behaviour it covers.
            if content.contains("TYPE|QUESTION|ANSWER") {
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
            generate_response:
                "recall|What was the exit code?|0\nartifact|What file was created?|src/auth.rs"
                    .into(),
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

#[cfg(test)]
mod parse_tolerance_tests {
    use super::*;

    struct NullCompleter;
    impl Completer for NullCompleter {
        fn complete(&self, _p: &str, _m: usize) -> crate::error::Result<String> {
            Ok(String::new())
        }
    }

    /// The exact output qwen2.5:3b produced against the original prompt. Every
    /// line was silently discarded, so the evaluator reported zero probes and
    /// the caller could not tell that from a context with nothing worth asking.
    #[test]
    fn numbered_list_with_padding_pipes_is_recovered() {
        let response = "\
Here are 4 factual questions based on the conversation:

1. |recall|What was the command executed at step 27?|ran command 27
2. |artifact|Which file was modified?|src/auth.rs
- decision|Why was Postgres chosen?|for the JSONB support
";
        let probes = ProbeEvaluator::<NullCompleter>::parse_probes(response).unwrap();
        assert_eq!(probes.len(), 3, "got {probes:?}");
        assert_eq!(probes[0].probe_type, ProbeType::Recall);
        assert_eq!(probes[1].probe_type, ProbeType::Artifact);
        assert_eq!(probes[2].probe_type, ProbeType::Decision);
        assert_eq!(probes[1].question, "Which file was modified?");
        assert_eq!(probes[1].expected, "src/auth.rs");
    }

    #[test]
    fn plain_format_still_parses_and_prose_is_ignored() {
        let response = "\
recall|What failed?|the build
this line is prose and must not become a probe
continuation|What is next?|fix the type error
";
        let probes = ProbeEvaluator::<NullCompleter>::parse_probes(response).unwrap();
        assert_eq!(probes.len(), 2);
        assert_eq!(probes[0].probe_type, ProbeType::Recall);
        assert_eq!(probes[1].probe_type, ProbeType::Continuation);
    }
}

#[cfg(test)]
mod robustness_tests {
    use super::*;

    struct Canned(String);
    impl Completer for Canned {
        fn complete(&self, _p: &str, _m: usize) -> crate::error::Result<String> {
            Ok(self.0.clone())
        }
    }

    #[test]
    fn no_parseable_probes_is_an_error_not_an_empty_success() {
        // A judge that ignores the format used to yield Ok(vec![]), which a
        // caller cannot distinguish from a context with nothing worth asking.
        let ev = ProbeEvaluator::new(Canned(
            "Sure! Here are some questions about the conversation.".into(),
        ));
        let msgs = vec![Message::user("hello")];
        match ev.generate_probes(&msgs) {
            Err(crate::error::Error::NoProbesParsed { lines }) => assert_eq!(lines, 1),
            other => panic!("expected NoProbesParsed, got {other:?}"),
        }
    }

    #[test]
    fn long_context_keeps_head_and_tail() {
        let ev = ProbeEvaluator::new(Canned(String::new())).max_context_chars(400);
        let msgs = vec![
            Message::user("FIRST_MARKER open the project"),
            Message::assistant("x".repeat(5_000)),
            Message::assistant("LAST_MARKER done"),
        ];
        let rendered = ev.render(&msgs);
        assert!(rendered.len() < 1_000, "len was {}", rendered.len());
        // Both ends must survive: probes drawn only from the opening would ask
        // about content every compressor keeps, flattering all of them.
        assert!(rendered.contains("FIRST_MARKER"));
        assert!(rendered.contains("LAST_MARKER"));
        assert!(rendered.contains("omitted"));
    }

    #[test]
    fn short_context_is_untouched() {
        let ev = ProbeEvaluator::new(Canned(String::new()));
        let msgs = vec![Message::user("small"), Message::assistant("also small")];
        let rendered = ev.render(&msgs);
        assert!(!rendered.contains("omitted"));
        assert!(rendered.contains("small"));
    }

    #[test]
    fn multibyte_context_does_not_panic_at_the_cut() {
        // Slicing on a byte offset inside a multi-byte character panics.
        let ev = ProbeEvaluator::new(Canned(String::new())).max_context_chars(120);
        let msgs = vec![
            Message::user("é".repeat(500)),
            Message::assistant("日本語テキスト".repeat(200)),
        ];
        let rendered = ev.render(&msgs);
        assert!(rendered.contains("omitted"));
    }
}
