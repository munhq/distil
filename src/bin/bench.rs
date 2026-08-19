//! `distil-bench` — measure where tokens actually go in real agent transcripts.
//!
//! This is the denominator every compression claim needs and none of them
//! publish. A tool that reports "92% saved" has told you nothing until you know
//! what share of a real session it was allowed to touch: if tool results are a
//! third of the tokens, no tool-result compressor can save more than a third.
//!
//! Two numbers are produced side by side:
//!
//! 1. A local tiktoken count per segment class, which says where the text is.
//! 2. The provider's own billed `usage` totals, which say what was paid.
//!
//! They answer different questions and are both reported rather than blended.
//! The billed figure includes the system prompt and tool schemas, which never
//! appear in a transcript; the local count covers only what the file contains.
//! The gap between them is itself a finding, so it is printed, not hidden.
//!
//! Usage:
//!   distil-bench <corpus-dir> [--limit N] [--json out.json] [--model NAME]

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use rayon::prelude::*;

use distil::corpus::{find_transcripts, load_session, SegmentKind, TurnUsage};
use distil::counter::{counter_for_model, TokenCounter};

/// Per-class totals for one pass over the corpus.
#[derive(Default, Clone)]
struct Totals {
    /// Local token count, keyed by segment class.
    by_kind: HashMap<&'static str, u64>,
    /// Segment occurrences, keyed by class.
    count_by_kind: HashMap<&'static str, u64>,
    /// Local token count, keyed by tool name.
    by_tool: HashMap<String, u64>,
    /// Tool-result occurrences, keyed by tool name.
    calls_by_tool: HashMap<String, u64>,
    /// Local token count of tool OUTPUT, keyed by tool name. This is the large
    /// half: arguments are a request, results are a payload.
    result_by_tool: HashMap<String, u64>,
    result_calls_by_tool: HashMap<String, u64>,
    billed: TurnUsage,
    sessions: u64,
    /// Sessions holding at least one assistant turn.
    non_empty_sessions: u64,
    assistant_turns: u64,
    malformed_lines: u64,
    /// Per-session local totals, retained for percentile reporting.
    session_tokens: Vec<u64>,
    /// Per-session assistant-turn counts. Compression pays only when turns
    /// remain, so the shape of this distribution bounds the whole opportunity.
    session_turns: Vec<u64>,
}

impl Totals {
    fn merge(mut self, other: Totals) -> Totals {
        for (k, v) in other.by_kind {
            *self.by_kind.entry(k).or_insert(0) += v;
        }
        for (k, v) in other.count_by_kind {
            *self.count_by_kind.entry(k).or_insert(0) += v;
        }
        for (k, v) in other.by_tool {
            *self.by_tool.entry(k).or_insert(0) += v;
        }
        for (k, v) in other.calls_by_tool {
            *self.calls_by_tool.entry(k).or_insert(0) += v;
        }
        for (k, v) in other.result_by_tool {
            *self.result_by_tool.entry(k).or_insert(0) += v;
        }
        for (k, v) in other.result_calls_by_tool {
            *self.result_calls_by_tool.entry(k).or_insert(0) += v;
        }
        self.billed.input += other.billed.input;
        self.billed.cache_creation += other.billed.cache_creation;
        self.billed.cache_read += other.billed.cache_read;
        self.billed.output += other.billed.output;
        self.billed.cache_write_5m += other.billed.cache_write_5m;
        self.billed.cache_write_1h += other.billed.cache_write_1h;
        self.sessions += other.sessions;
        self.non_empty_sessions += other.non_empty_sessions;
        self.assistant_turns += other.assistant_turns;
        self.malformed_lines += other.malformed_lines;
        self.session_tokens.extend(other.session_tokens);
        self.session_turns.extend(other.session_turns);
        self
    }

    fn local_total(&self) -> u64 {
        self.by_kind.values().sum()
    }
}

/// Measure one session. `split_prefix` marks the session as belonging to the
/// treatment group when any tool it CALLED starts with that prefix.
///
/// Matching on calls, not on text, is deliberate: a tool name also appears in
/// system reminders and prose, so a grep over the file finds sessions that
/// merely had the tool available. Availability is not use.
fn measure_one(
    path: &Path,
    counter: &dyn TokenCounter,
    split_prefix: Option<&str>,
) -> (Totals, bool) {
    let mut t = Totals {
        sessions: 1,
        ..Default::default()
    };
    let session = match load_session(path) {
        Ok(s) => s,
        // An unreadable file is skipped rather than aborting a 13,000-file run.
        Err(_) => return (t, false),
    };
    let mut used = false;

    t.malformed_lines = session.malformed_lines as u64;
    t.assistant_turns = session.assistant_turns as u64;
    if session.assistant_turns > 0 {
        t.non_empty_sessions = 1;
    }
    t.billed = session.billed();

    let mut session_total = 0u64;
    for seg in &session.segments {
        // An image has no text; counting it as zero tokens would understate it,
        // so it is tracked by occurrence only and excluded from token sums.
        let tokens = if seg.kind == SegmentKind::Image {
            0
        } else {
            counter.count(&seg.text) as u64
        };
        *t.by_kind.entry(seg.kind.as_str()).or_insert(0) += tokens;
        *t.count_by_kind.entry(seg.kind.as_str()).or_insert(0) += 1;
        session_total += tokens;

        if seg.kind == SegmentKind::ToolUse {
            if let Some(name) = &seg.tool {
                if let Some(pfx) = split_prefix {
                    if name.starts_with(pfx) {
                        used = true;
                    }
                }
                *t.by_tool.entry(name.clone()).or_insert(0) += tokens;
                *t.calls_by_tool.entry(name.clone()).or_insert(0) += 1;
            }
        }
        if seg.kind == SegmentKind::ToolResult {
            if let Some(name) = &seg.tool {
                *t.result_by_tool.entry(name.clone()).or_insert(0) += tokens;
                *t.result_calls_by_tool.entry(name.clone()).or_insert(0) += 1;
            }
        }
    }
    t.session_tokens.push(session_total);
    if session.assistant_turns > 0 {
        t.session_turns.push(session.assistant_turns as u64);
    }
    (t, used)
}

fn pct(part: u64, whole: u64) -> f64 {
    if whole == 0 {
        0.0
    } else {
        part as f64 * 100.0 / whole as f64
    }
}

fn percentile(sorted: &[u64], p: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[idx]
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: distil-bench <corpus-dir> [--limit N] [--json out.json] [--model NAME]");
        std::process::exit(2);
    }
    let root = PathBuf::from(&args[1]);
    let flag = |name: &str| -> Option<String> {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let limit: Option<usize> = flag("--limit").and_then(|v| v.parse().ok());
    let json_out = flag("--json");
    let model = flag("--model").unwrap_or_else(|| "claude-opus-4".to_string());
    let split_by = flag("--split-by-tool");
    // Export real tool results so external compressors can be run on the same
    // input. A compressor benchmarked on its own fixtures proves nothing.
    let export = flag("--export-payloads");
    let export_sessions = flag("--export-sessions");
    let min_turns: usize = flag("--min-turns").and_then(|v| v.parse().ok()).unwrap_or(40);
    let max_sessions: usize = flag("--max-sessions").and_then(|v| v.parse().ok()).unwrap_or(60);
    let per_tool_cap: usize = flag("--per-tool")
        .and_then(|v| v.parse().ok())
        .unwrap_or(400);

    let mut files = match find_transcripts(&root) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("cannot read corpus at {}: {e}", root.display());
            std::process::exit(1);
        }
    };
    let found = files.len();
    if let Some(n) = limit {
        files.truncate(n);
    }
    eprintln!(
        "corpus: {} transcripts found, {} selected, model={}",
        found,
        files.len(),
        model
    );

    let counter = counter_for_model(&model);
    let done = AtomicUsize::new(0);
    let step = (files.len() / 20).max(1);

    let measured: Vec<(Totals, bool)> = files
        .par_iter()
        .map(|p| {
            let t = measure_one(p, counter.as_ref(), split_by.as_deref());
            let n = done.fetch_add(1, Ordering::Relaxed) + 1;
            if n % step == 0 {
                eprintln!("  {n}/{} sessions", files.len());
            }
            t
        })
        .collect();

    if let Some(pfx) = &split_by {
        report_split(&measured, pfx);
        return;
    }

    if let Some(out) = &export {
        export_payloads(&files, counter.as_ref(), out, per_tool_cap);
        return;
    }

    if let Some(dir) = &export_sessions {
        export_sessions_to(&files, dir, min_turns, max_sessions);
        return;
    }

    let totals = measured
        .into_iter()
        .map(|(t, _)| t)
        .fold(Totals::default(), Totals::merge);

    let local = totals.local_total();
    let billed_in = totals.billed.input_total();

    println!("\n=== corpus ===");
    println!("sessions scanned      {}", totals.sessions);
    println!(
        "sessions with content {} ({:.1}%)",
        totals.non_empty_sessions,
        pct(totals.non_empty_sessions, totals.sessions)
    );
    println!("assistant turns       {}", totals.assistant_turns);
    println!("malformed lines       {}", totals.malformed_lines);

    println!("\n=== where the tokens are (local tiktoken count) ===");
    println!(
        "{:<16} {:>14} {:>8} {:>12}",
        "class", "tokens", "share", "segments"
    );
    let mut kinds: Vec<_> = SegmentKind::ALL
        .iter()
        .map(|k| {
            let name = k.as_str();
            (
                name,
                *totals.by_kind.get(name).unwrap_or(&0),
                *totals.count_by_kind.get(name).unwrap_or(&0),
            )
        })
        .collect();
    kinds.sort_by_key(|(_, tok, _)| std::cmp::Reverse(*tok));
    for (name, tok, n) in &kinds {
        println!(
            "{:<16} {:>14} {:>7.1}% {:>12}",
            name,
            tok,
            pct(*tok, local),
            n
        );
    }
    println!("{:<16} {:>14} {:>7.1}%", "TOTAL", local, 100.0);

    println!("\n=== what the provider billed (tokens) ===");
    println!(
        "{:<22} {:>16} {:>8}",
        "class", "tokens", "share"
    );
    for (label, v) in [
        ("input (uncached)", totals.billed.input),
        ("cache write 5m", totals.billed.cache_write_5m),
        ("cache write 1h", totals.billed.cache_write_1h),
        ("cache read", totals.billed.cache_read),
    ] {
        println!("{label:<22} {v:>16} {:>7.1}%", pct(v, billed_in));
    }
    println!("{:<22} {billed_in:>16} {:>7.1}%", "input total", 100.0);
    println!("{:<22} {:>16}", "output", totals.billed.output);
    println!(
        "\nlocal transcript text is {:.2}% of billed input.",
        pct(local, billed_in)
    );
    println!(
        "amplification: every unique token was billed {:.0} times on average.",
        billed_in as f64 / local.max(1) as f64
    );
    println!("A transcript stores each message once; a request resends the whole history.");

    // Re-express input in multiples of the base input price. A raw token count
    // makes cache reads look dominant when they are billed at a tenth.
    let u_uncached = totals.billed.input as f64;
    let u_read = totals.billed.cache_read as f64 * 0.1;
    let u_w5 = totals.billed.cache_write_5m as f64 * 1.25;
    let u_w1h = totals.billed.cache_write_1h as f64 * 2.0;
    let u_in = u_uncached + u_read + u_w5 + u_w1h;

    println!("\n=== what it COST (input priced at its real multiplier) ===");
    println!("cache read  x0.1   cache write x1.25 (5m) / x2.0 (1h)   uncached x1.0");
    println!("{:<22} {:>16} {:>8}", "class", "price units", "share");
    for (label, v) in [
        ("input (uncached)", u_uncached),
        ("cache write 5m", u_w5),
        ("cache write 1h", u_w1h),
        ("cache read", u_read),
    ] {
        println!(
            "{label:<22} {:>16.0} {:>7.1}%",
            v,
            if u_in > 0.0 { v * 100.0 / u_in } else { 0.0 }
        );
    }
    println!("{:<22} {u_in:>16.0} {:>7.1}%", "input total", 100.0);

    let write_tokens = totals.billed.cache_write_5m + totals.billed.cache_write_1h;
    println!(
        "\ncache writes are {:.1}% of input TOKENS but {:.1}% of input COST.",
        pct(write_tokens, billed_in),
        if u_in > 0.0 {
            (u_w5 + u_w1h) * 100.0 / u_in
        } else {
            0.0
        }
    );
    println!("Any edit to the history invalidates the cached prefix from that point on,");
    println!("which turns cache reads at 0.1x into cache writes at 1.25x or 2.0x.");

    // Break-even for a history rewrite.
    //
    // Leaving history alone costs K * N * 0.1, because every one of the K
    // remaining turns re-reads N cached tokens at a tenth of base price.
    // Rewriting it to M tokens costs one write at W, then K-1 cheap reads:
    //     M * W + (K - 1) * M * 0.1  <  K * N * 0.1
    // Solving for the surviving fraction:
    //     M / N  <  0.1 * K / (W - 0.1 + 0.1 * K)
    //
    // The consequence is that compression is not a property of the context. It
    // is a property of how much conversation REMAINS. The same edit that pays
    // for itself at turn 5 of 60 is a loss at turn 58.
    fn max_surviving_fraction(remaining_turns: f64, write_mult: f64) -> f64 {
        0.1 * remaining_turns / (write_mult - 0.1 + 0.1 * remaining_turns)
    }

    println!("\n=== when a history rewrite pays for itself ===");
    println!("A rewrite must shrink history to below this fraction to break even:");
    println!(
        "{:<18} {:>18} {:>18}",
        "turns remaining", "5m TTL (x1.25)", "1h TTL (x2.0)"
    );
    for k in [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0] {
        println!(
            "{:<18} {:>17.1}% {:>17.1}%",
            k as u64,
            max_surviving_fraction(k, 1.25) * 100.0,
            max_surviving_fraction(k, 2.0) * 100.0
        );
    }
    println!("\nRead the first row: with one turn left, a rewrite must delete over 92%");
    println!("of the history merely to break even on price, before any quality loss.");
    println!("Compression is therefore a bet on the conversation continuing.");

    println!("\n=== top tools by RESULT tokens (what the tool put in context) ===");
    let mut rtools: Vec<_> = totals.result_by_tool.iter().collect();
    rtools.sort_by_key(|(_, v)| std::cmp::Reverse(**v));
    let result_total: u64 = totals.result_by_tool.values().sum();
    println!(
        "{:<34} {:>14} {:>8} {:>10} {:>10}",
        "tool", "result tokens", "share", "calls", "per call"
    );
    for (name, tok) in rtools.iter().take(18) {
        let calls = *totals.result_calls_by_tool.get(*name).unwrap_or(&0);
        println!(
            "{:<34} {:>14} {:>7.1}% {:>10} {:>10}",
            name,
            tok,
            pct(**tok, result_total),
            calls,
            if calls > 0 { **tok / calls } else { 0 }
        );
    }

    let mut turns = totals.session_turns.clone();
    turns.sort_unstable();
    println!("\n=== assistant turns per session ===");
    for (label, p) in [
        ("p50", 0.50),
        ("p75", 0.75),
        ("p90", 0.90),
        ("p99", 0.99),
        ("max", 1.00),
    ] {
        println!("{label:<6} {:>12}", percentile(&turns, p));
    }
    // A session's turns are also its compression opportunities: a rewrite at
    // turn i is amortised over the turns after it. Short sessions cannot repay
    // a cache invalidation at any compression ratio, so they are counted out.
    let long = turns.iter().filter(|&&t| t >= 20).count();
    let turn_total: u64 = turns.iter().sum();
    let turns_in_long: u64 = turns.iter().filter(|&&t| t >= 20).sum();
    println!(
        "\nsessions with 20+ turns: {} of {} ({:.1}%)",
        long,
        turns.len(),
        pct(long as u64, turns.len() as u64)
    );
    println!(
        "but they hold {:.1}% of all assistant turns.",
        pct(turns_in_long, turn_total)
    );
    println!("Compression can only pay inside those; the rest are too short to amortise a rewrite.");

    let mut sorted = totals.session_tokens.clone();
    sorted.sort_unstable();
    println!("\n=== session size (local tokens) ===");
    for (label, p) in [
        ("p50", 0.50),
        ("p75", 0.75),
        ("p90", 0.90),
        ("p99", 0.99),
        ("max", 1.00),
    ] {
        println!("{label:<6} {:>12}", percentile(&sorted, p));
    }

    if let Some(path) = json_out {
        let doc = serde_json::json!({
            "model": model,
            "sessions_scanned": totals.sessions,
            "sessions_with_content": totals.non_empty_sessions,
            "assistant_turns": totals.assistant_turns,
            "malformed_lines": totals.malformed_lines,
            "local_tokens_by_class": totals.by_kind,
            "segments_by_class": totals.count_by_kind,
            "local_total": local,
            "billed": {
                "input": totals.billed.input,
                "cache_creation": totals.billed.cache_creation,
                "cache_read": totals.billed.cache_read,
                "cache_write_5m": totals.billed.cache_write_5m,
                "cache_write_1h": totals.billed.cache_write_1h,
                "input_total": billed_in,
                "output": totals.billed.output,
            },
            "input_price_units": {
                "uncached": u_uncached,
                "cache_write_5m": u_w5,
                "cache_write_1h": u_w1h,
                "cache_read": u_read,
                "total": u_in,
            },
            "tool_arg_tokens": totals.by_tool,
            "tool_calls": totals.calls_by_tool,
            "turns_p50": percentile(&turns, 0.50),
            "turns_p90": percentile(&turns, 0.90),
            "turns_p99": percentile(&turns, 0.99),
            "sessions_20plus_turns": long,
            "turn_share_in_20plus": pct(turns_in_long, turn_total),
            "session_percentiles": {
                "p50": percentile(&sorted, 0.50),
                "p75": percentile(&sorted, 0.75),
                "p90": percentile(&sorted, 0.90),
                "p99": percentile(&sorted, 0.99),
                "max": percentile(&sorted, 1.00),
            },
        });
        match std::fs::write(&path, serde_json::to_string_pretty(&doc).unwrap()) {
            Ok(_) => eprintln!("\nwrote {path}"),
            Err(e) => eprintln!("\ncannot write {path}: {e}"),
        }
    }
}

/// Compare sessions that CALLED a tool against those that did not.
///
/// This is observational, not a randomised trial. Sessions reach for a tool
/// because of what they are doing, so a difference between the groups is not
/// by itself an effect of the tool. Two guards are applied and both are
/// reported: figures are normalised per assistant turn rather than per session,
/// and the groups are also compared inside matched turn-count bands, because
/// session length is the confounder that swamps everything else.
fn report_split(measured: &[(Totals, bool)], prefix: &str) {
    let fold = |want: bool| -> Totals {
        measured
            .iter()
            .filter(|(t, used)| *used == want && t.assistant_turns > 0)
            .map(|(t, _)| t.clone())
            .fold(Totals::default(), Totals::merge)
    };
    let with = fold(true);
    let without = fold(false);

    println!("\n=== A/B split on tool prefix `{prefix}` ===");
    println!("Grouped by whether the session actually CALLED a matching tool.");
    println!(
        "\n{:<26} {:>16} {:>16} {:>10}",
        "metric", "with", "without", "delta"
    );

    let row = |label: &str, a: f64, b: f64| {
        let delta = if b.abs() > f64::EPSILON {
            (a - b) / b * 100.0
        } else {
            0.0
        };
        println!("{label:<26} {a:>16.1} {b:>16.1} {delta:>9.1}%");
    };

    let per_turn = |t: &Totals, key: &str| -> f64 {
        if t.assistant_turns == 0 {
            return 0.0;
        }
        *t.by_kind.get(key).unwrap_or(&0) as f64 / t.assistant_turns as f64
    };

    println!("sessions                   {:>16} {:>16}", 
        with.session_turns.len(), without.session_turns.len());
    println!("assistant turns            {:>16} {:>16}",
        with.assistant_turns, without.assistant_turns);

    row("tool_result tok/turn", per_turn(&with, "tool_result"), per_turn(&without, "tool_result"));
    row("tool_use tok/turn", per_turn(&with, "tool_use"), per_turn(&without, "tool_use"));
    row("assistant_text tok/turn", per_turn(&with, "assistant_text"), per_turn(&without, "assistant_text"));
    row(
        "all text tok/turn",
        with.local_total() as f64 / with.assistant_turns.max(1) as f64,
        without.local_total() as f64 / without.assistant_turns.max(1) as f64,
    );
    row(
        "billed input units/turn",
        with.billed.billable_input_units() / with.assistant_turns.max(1) as f64,
        without.billed.billable_input_units() / without.assistant_turns.max(1) as f64,
    );
    row(
        "output tok/turn",
        with.billed.output as f64 / with.assistant_turns.max(1) as f64,
        without.billed.output as f64 / without.assistant_turns.max(1) as f64,
    );

    // Session length is the dominant confounder: long sessions carry more
    // history per turn regardless of which tools they used. Comparing inside
    // bands holds it roughly fixed.
    println!("\n--- same comparison inside matched session-length bands ---");
    println!(
        "{:<18} {:>10} {:>10} {:>14} {:>14} {:>9}",
        "turns band", "n with", "n without", "with tok/turn", "w/o tok/turn", "delta"
    );
    for (label, lo, hi) in [
        ("1-9", 1u64, 9u64),
        ("10-49", 10, 49),
        ("50-199", 50, 199),
        ("200+", 200, u64::MAX),
    ] {
        let band = |want: bool| -> (usize, f64) {
            let sel: Vec<&Totals> = measured
                .iter()
                .filter(|(t, used)| {
                    *used == want
                        && t.assistant_turns >= lo
                        && t.assistant_turns <= hi
                })
                .map(|(t, _)| t)
                .collect();
            let turns: u64 = sel.iter().map(|t| t.assistant_turns).sum();
            let toks: u64 = sel.iter().map(|t| t.local_total()).sum();
            (
                sel.len(),
                if turns > 0 {
                    toks as f64 / turns as f64
                } else {
                    0.0
                },
            )
        };
        let (nw, tw) = band(true);
        let (no, to) = band(false);
        let delta = if to.abs() > f64::EPSILON {
            (tw - to) / to * 100.0
        } else {
            0.0
        };
        println!("{label:<18} {nw:>10} {no:>10} {tw:>14.1} {to:>14.1} {delta:>8.1}%");
    }

    println!("\n--- what the tool itself put in context ---");
    let mut rt: Vec<_> = with.result_by_tool.iter().collect();
    rt.sort_by_key(|(_, v)| std::cmp::Reverse(**v));
    println!("{:<34} {:>14} {:>10} {:>10}", "tool", "result tokens", "calls", "per call");
    for (name, tok) in rt.iter().take(12) {
        let calls = *with.result_calls_by_tool.get(*name).unwrap_or(&0);
        println!(
            "{:<34} {:>14} {:>10} {:>10}",
            name, tok, calls,
            if calls > 0 { **tok / calls } else { 0 }
        );
    }
}

/// Write real tool-result payloads to JSONL so external compressors can run on
/// the same input this crate measured.
///
/// The sample is stratified: at most `cap` payloads per tool. Tool output is
/// extremely skewed — `Read` alone is most of the corpus — and an unstratified
/// sample would measure one tool while claiming to measure the corpus. The true
/// population weight of each tool is written alongside, so a result can be
/// reweighted back to corpus proportions instead of reported as a raw mean.
fn export_payloads(
    files: &[PathBuf],
    counter: &dyn TokenCounter,
    out_path: &str,
    cap: usize,
) {
    use std::io::Write;

    // Population totals first, so weights are exact rather than estimated.
    let pop: HashMap<String, (u64, u64)> = files
        .par_iter()
        .map(|p| {
            let mut m: HashMap<String, (u64, u64)> = HashMap::new();
            if let Ok(s) = load_session(p) {
                for seg in &s.segments {
                    if seg.kind != SegmentKind::ToolResult {
                        continue;
                    }
                    if let Some(t) = &seg.tool {
                        let e = m.entry(t.clone()).or_insert((0, 0));
                        e.0 += counter.count(&seg.text) as u64;
                        e.1 += 1;
                    }
                }
            }
            m
        })
        .reduce(HashMap::new, |mut a, b| {
            for (k, (tok, n)) in b {
                let e = a.entry(k).or_insert((0, 0));
                e.0 += tok;
                e.1 += n;
            }
            a
        });

    let mut file = match std::fs::File::create(out_path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("cannot write {out_path}: {e}");
            return;
        }
    };

    let mut taken: HashMap<String, usize> = HashMap::new();
    let mut written = 0usize;
    // Sequential so the per-tool cap is deterministic and reproducible.
    for p in files {
        let s = match load_session(p) {
            Ok(s) => s,
            Err(_) => continue,
        };
        for seg in &s.segments {
            if seg.kind != SegmentKind::ToolResult || seg.text.is_empty() {
                continue;
            }
            let tool = match &seg.tool {
                Some(t) => t.clone(),
                None => continue,
            };
            let c = taken.entry(tool.clone()).or_insert(0);
            if *c >= cap {
                continue;
            }
            *c += 1;
            let (pop_tokens, pop_calls) =
                pop.get(&tool).copied().unwrap_or((0, 0));
            let rec = serde_json::json!({
                "tool": tool,
                "tokens": counter.count(&seg.text),
                "text": seg.text,
                "pop_tokens": pop_tokens,
                "pop_calls": pop_calls,
            });
            if writeln!(file, "{rec}").is_err() {
                eprintln!("write failed at record {written}");
                return;
            }
            written += 1;
        }
    }
    eprintln!("exported {written} payloads across {} tools to {out_path}", taken.len());
}

/// Write whole sessions as Anthropic-format message arrays.
///
/// A context compressor is a function of the WHOLE conversation, not of one
/// payload. Headroom protects the last four messages and skips anything under
/// its minimum size; RTK acts at capture time. Feeding either an isolated
/// payload measures the harness, not the tool. Only sessions long enough to
/// have a compressible prefix are written.
fn export_sessions_to(files: &[PathBuf], dir: &str, min_turns: usize, max_sessions: usize) {
    if let Err(e) = std::fs::create_dir_all(dir) {
        eprintln!("cannot create {dir}: {e}");
        return;
    }
    let mut written = 0usize;
    for p in files {
        if written >= max_sessions {
            break;
        }
        let s = match load_session(p) {
            Ok(s) => s,
            Err(_) => continue,
        };
        if s.assistant_turns < min_turns || s.raw_messages.is_empty() {
            continue;
        }
        let name = p
            .file_stem()
            .and_then(|x| x.to_str())
            .unwrap_or("session")
            .to_string();
        let out = format!("{dir}/{name}.json");
        let body = match serde_json::to_string(&s.raw_messages) {
            Ok(b) => b,
            Err(_) => continue,
        };
        if std::fs::write(&out, body).is_ok() {
            written += 1;
        }
    }
    eprintln!("exported {written} sessions with >= {min_turns} assistant turns to {dir}");
}
