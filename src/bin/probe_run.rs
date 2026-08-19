//! `distil-probe` — does compression cost the model information?
//!
//! Everything else in this crate measures what compression COSTS. This measures
//! what it BREAKS. A saving is only a saving if the model can still answer the
//! questions the original context could answer.
//!
//! The method is Factory.ai's four probe types — recall, artifact, continuation
//! and decision (https://factory.ai/news/evaluating-compression). Their write-up
//! defines the taxonomy and reports that every compressor they tested scores
//! 2.19-2.45 out of 5.0 on artifact tracking; it ships no harness. This is one.
//!
//! An LLM generates the probes from the ORIGINAL context and grades answers
//! drawn from the COMPRESSED context. That makes the judge a dependency, and a
//! small local judge is a weak one — treat the absolute rate as indicative and
//! the comparison between conditions as the real output. For a metric with no
//! judge at all, see `bench/artifact_retention.py`, which checks file paths
//! against ground truth.
//!
//! The judge is a [`distil::Completer`], never a `Summarizer`: a summarizer is
//! entitled to wrap its input in summarization framing, which rewrites both the
//! probe format and the grading instruction. Swap `OllamaCompleter` for a
//! hosted-model `Completer` to run this against a judge worth quoting.
//!
//! Usage:
//!   distil-probe <session.jsonl> [--probes N] [--model NAME] [--retain N]

use distil::corpus::load_session;
use distil::counter::{TokenCounter, counter_for_model};
use distil::layers::MaskingLayer;
use distil::pipeline::{Ctx, Pipeline};
use distil::probe::ProbeEvaluator;
use distil::summarizer::OllamaCompleter;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: distil-probe <session.jsonl> [--probes N] [--model NAME] [--retain N]");
        std::process::exit(2);
    }
    let path = std::path::PathBuf::from(&args[1]);
    let flag = |n: &str| -> Option<String> {
        args.iter()
            .position(|a| a == n)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let n_probes: usize = flag("--probes").and_then(|v| v.parse().ok()).unwrap_or(6);
    let model = flag("--model").unwrap_or_else(|| "qwen2.5:3b".to_string());
    let retain: u32 = flag("--retain").and_then(|v| v.parse().ok()).unwrap_or(3);

    let session = match load_session(&path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("cannot read {}: {e}", path.display());
            std::process::exit(1);
        }
    };
    let original = session.to_messages();
    if original.is_empty() {
        eprintln!("session has no messages");
        std::process::exit(1);
    }

    let counter = counter_for_model("claude-opus-4");
    let before: usize = original.iter().map(|m| counter.count(&m.content)).sum();

    // Compress with distil's own masking layer, which is the layer that would
    // actually be shipped. The point is to grade a real transformation, not a
    // synthetic one.
    let mut ctx = Ctx::new(original.clone(), vec![], session.assistant_turns as u32);
    let pipeline = Pipeline::builder()
        .counter(counter_for_model("claude-opus-4"))
        .layer(MaskingLayer::new().retain_turns(retain))
        .build();
    let result = pipeline.optimize(&mut ctx);
    let after: usize = ctx.messages.iter().map(|m| counter.count(&m.content)).sum();

    println!("session   {}", path.file_name().unwrap().to_string_lossy());
    println!("messages  {}", original.len());
    println!(
        "tokens    {before} -> {after}  ({:.1}% saved)",
        if before > 0 {
            (before - after) as f64 * 100.0 / before as f64
        } else {
            0.0
        }
    );
    println!("pipeline  {result}");

    println!("\njudge: ollama/{model} (small local model — see the module note)");
    let timeout: u64 = flag("--timeout")
        .and_then(|v| v.parse().ok())
        .unwrap_or(600);
    let evaluator = ProbeEvaluator::new(OllamaCompleter::new(&model, timeout)).num_probes(n_probes);

    let probes = match evaluator.generate_probes(&original) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("probe generation failed: {e}");
            std::process::exit(1);
        }
    };
    println!(
        "generated {} probes from the ORIGINAL context",
        probes.len()
    );
    for p in &probes {
        println!("  [{}] {}", p.probe_type, p.question);
    }
    if probes.is_empty() {
        eprintln!("\nno probes parsed — the judge did not follow the output format");
        std::process::exit(1);
    }

    // Grade the compressed context, then the original as a control. Without the
    // control an unanswerable probe looks like compression damage.
    let compressed_report = match evaluator.evaluate_probes(&ctx.messages, &probes) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("evaluation failed: {e}");
            std::process::exit(1);
        }
    };
    let control_report = evaluator.evaluate_probes(&original, &probes).ok();

    println!("\n--- COMPRESSED context ---");
    print!("{compressed_report}");
    if let Some(c) = &control_report {
        println!("--- ORIGINAL context (control) ---");
        print!("{c}");
        let delta = compressed_report.success_rate - c.success_rate;
        println!("\nretention delta vs control: {:+.1} points", delta * 100.0);
        println!(
            "A control below 100% is judge error, not compression damage. Only the\n\
             gap between the two lines is attributable to compression."
        );
    }
}
