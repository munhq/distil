/// Trait for counting tokens in a string.
///
/// Implement this to plug in your own tokenizer (tiktoken, sentencepiece, etc.).
/// The default [`EstimateCounter`] uses a chars/3.5 heuristic that's reasonable
/// for English text and code. For billing-accurate counts, use [`TiktokenCounter`]
/// (requires the `tiktoken` feature).
pub trait TokenCounter: Send + Sync {
    fn count(&self, text: &str) -> usize;
}

impl TokenCounter for Box<dyn TokenCounter> {
    fn count(&self, text: &str) -> usize {
        (**self).count(text)
    }
}

/// Returns the best available counter for a given model name.
///
/// With the `tiktoken` feature enabled, this returns a [`TiktokenCounter`]
/// matched to the model's encoding. Falls back to [`EstimateCounter`] for
/// unknown models or when the feature is disabled.
///
/// # Examples
/// ```rust,ignore
/// let counter = distil::counter::counter_for_model("gpt-4o");
/// ```
pub fn counter_for_model(model: &str) -> Box<dyn TokenCounter> {
    #[cfg(feature = "tiktoken")]
    if let Some(c) = TiktokenCounter::for_model(model) {
        return Box::new(c);
    }
    let _ = model;
    Box::new(EstimateCounter)
}

/// Estimates tokens as `ceil(chars / 3.5)`.
///
/// This is the industry-standard heuristic: English text averages ~4 chars/token,
/// code averages ~3 chars/token, 3.5 splits the difference. Good enough for
/// budget decisions; use a real tokenizer for billing accuracy.
#[derive(Debug, Clone, Copy, Default)]
pub struct EstimateCounter;

impl TokenCounter for EstimateCounter {
    fn count(&self, text: &str) -> usize {
        if text.is_empty() {
            return 0;
        }
        (text.len() as f64 / 3.5).ceil() as usize
    }
}

/// Counts tokens by splitting on whitespace. Slightly more accurate than
/// char-based for natural language, less accurate for code.
#[derive(Debug, Clone, Copy, Default)]
pub struct WordCounter;

impl TokenCounter for WordCounter {
    fn count(&self, text: &str) -> usize {
        if text.is_empty() {
            return 0;
        }
        // Roughly 1.3 tokens per whitespace-delimited word
        let words = text.split_whitespace().count();
        ((words as f64) * 1.3).ceil() as usize
    }
}

/// Accurate BPE token counter backed by tiktoken.
///
/// Uses the same tokenizer as OpenAI's API. For Claude models, `cl100k_base`
/// (GPT-4's encoding) is used — Claude's BPE vocabulary is similar enough that
/// counts are within ~2-5% of the real value.
///
/// Requires the `tiktoken` feature flag.
///
/// # Model mapping
/// | Pattern | Encoding |
/// |---------|----------|
/// | `gpt-4*`, `gpt-4o*`, `claude-*`, `text-embedding-*` | `cl100k_base` |
/// | `gpt-3.5-turbo*` | `cl100k_base` |
/// | `llama*`, `mistral*`, `qwen*`, unknown | `cl100k_base` (best approximation) |
#[cfg(feature = "tiktoken")]
pub struct TiktokenCounter {
    bpe: tiktoken_rs::CoreBPE,
    encoding_name: &'static str,
}

#[cfg(feature = "tiktoken")]
impl TiktokenCounter {
    /// Create a counter using the encoding for a specific model name.
    ///
    /// Falls back to `cl100k_base` for unrecognised models.
    /// Returns `None` only if the BPE data files are corrupt (never happens in practice).
    pub fn for_model(model: &str) -> Option<Self> {
        let (bpe, name) = Self::encoding_for_model(model)?;
        Some(Self { bpe, encoding_name: name })
    }

    /// Create a counter using `cl100k_base` (GPT-4 / Claude).
    /// Returns `None` only if BPE data files are corrupt (never happens in practice).
    pub fn cl100k() -> Option<Self> {
        Some(Self {
            bpe: tiktoken_rs::cl100k_base().ok()?,
            encoding_name: "cl100k_base",
        })
    }

    /// The encoding name this counter is using (e.g. `"cl100k_base"`).
    pub fn encoding_name(&self) -> &str {
        self.encoding_name
    }

    fn encoding_for_model(model: &str) -> Option<(tiktoken_rs::CoreBPE, &'static str)> {
        let m = model.to_lowercase();
        // p50k_base: older OpenAI completion models
        if m.starts_with("text-davinci-00")
            || m.starts_with("code-davinci-00")
            || m == "davinci"
            || m == "curie"
            || m == "babbage"
            || m == "ada"
        {
            return Some((tiktoken_rs::p50k_base().ok()?, "p50k_base"));
        }
        // cl100k_base: everything modern — GPT-4, GPT-3.5, Claude, embeddings,
        // Llama/Mistral/Qwen all approximate well with this encoding.
        Some((tiktoken_rs::cl100k_base().ok()?, "cl100k_base"))
    }
}

#[cfg(feature = "tiktoken")]
impl TokenCounter for TiktokenCounter {
    fn count(&self, text: &str) -> usize {
        if text.is_empty() {
            return 0;
        }
        self.bpe.encode_with_special_tokens(text).len()
    }
}

#[cfg(feature = "tiktoken")]
impl std::fmt::Debug for TiktokenCounter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TiktokenCounter")
            .field("encoding", &self.encoding_name)
            .finish()
    }
}

/// Wraps a function as a [`TokenCounter`].
pub struct FnCounter<F>(pub F);

impl<F: Fn(&str) -> usize + Send + Sync> TokenCounter for FnCounter<F> {
    fn count(&self, text: &str) -> usize {
        (self.0)(text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimate_counter_empty() {
        assert_eq!(EstimateCounter.count(""), 0);
    }

    #[test]
    fn estimate_counter_short() {
        // "hello" = 5 chars → ceil(5/3.5) = 2
        assert_eq!(EstimateCounter.count("hello"), 2);
    }

    #[test]
    fn estimate_counter_paragraph() {
        let text = "The quick brown fox jumps over the lazy dog.";
        let tokens = EstimateCounter.count(text);
        // 44 chars → ceil(44/3.5) = 13, actual GPT-4 tokenization ≈ 10
        assert!(tokens > 5 && tokens < 20);
    }

    #[test]
    fn word_counter_empty() {
        assert_eq!(WordCounter.count(""), 0);
    }

    #[test]
    fn word_counter_sentence() {
        let text = "The quick brown fox jumps over the lazy dog.";
        let tokens = WordCounter.count(text);
        // 9 words * 1.3 = 11.7 → 12
        assert_eq!(tokens, 12);
    }

    #[test]
    fn fn_counter_works() {
        let counter = FnCounter(|s: &str| s.len());
        assert_eq!(counter.count("hello"), 5);
    }

    #[cfg(feature = "tiktoken")]
    mod tiktoken_tests {
        use super::*;

        #[test]
        fn tiktoken_empty() {
            let c = TiktokenCounter::cl100k().expect("cl100k_base failed");
            assert_eq!(c.count(""), 0);
        }

        #[test]
        fn tiktoken_hello_world() {
            let c = TiktokenCounter::cl100k().expect("cl100k_base failed");
            // "Hello, world!" → GPT-4 tokenizes as ["Hello", ",", " world", "!"] = 4 tokens
            let n = c.count("Hello, world!");
            assert_eq!(n, 4);
        }

        #[test]
        fn tiktoken_more_accurate_than_estimate() {
            let text = "The quick brown fox jumps over the lazy dog.";
            let tiktoken = TiktokenCounter::cl100k().expect("cl100k_base failed").count(text);
            let estimate = EstimateCounter.count(text);
            // tiktoken: 10 tokens (known ground truth for this sentence)
            assert_eq!(tiktoken, 10);
            // estimate overshoots
            assert!(estimate > tiktoken);
        }

        #[test]
        fn tiktoken_for_model_gpt4() {
            let c = TiktokenCounter::for_model("gpt-4o").expect("gpt-4o failed");
            assert_eq!(c.encoding_name(), "cl100k_base");
        }

        #[test]
        fn tiktoken_for_model_claude() {
            let c = TiktokenCounter::for_model("claude-opus-4-6").expect("claude failed");
            assert_eq!(c.encoding_name(), "cl100k_base");
        }

        #[test]
        fn tiktoken_for_model_unknown_falls_back_to_cl100k() {
            let c = TiktokenCounter::for_model("llama-3-70b").expect("llama failed");
            assert_eq!(c.encoding_name(), "cl100k_base");
        }

        #[test]
        fn tiktoken_for_model_old_completion() {
            let c = TiktokenCounter::for_model("text-davinci-003").expect("davinci failed");
            assert_eq!(c.encoding_name(), "p50k_base");
        }

        #[test]
        fn counter_for_model_returns_tiktoken() {
            let c = counter_for_model("gpt-4o");
            // Should count "Hello, world!" as 4 tokens via tiktoken
            assert_eq!(c.count("Hello, world!"), 4);
        }
    }
}
