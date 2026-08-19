//! Real agent-transcript corpus loader.
//!
//! Every published context-compression number is measured on synthetic or
//! curated tasks. This module reads the transcripts an agent actually wrote to
//! disk, so a measurement has a real denominator.
//!
//! Claude Code writes one JSONL file per session under
//! `~/.claude/projects/<project-hash>/<session-uuid>.jsonl`. One line is one
//! event. Most event types are harness bookkeeping (`attachment`,
//! `queue-operation`, `file-history-snapshot`, …); only `user` and `assistant`
//! lines carry conversation content, and only those are loaded.
//!
//! `progress` lines are DELIBERATELY skipped. They mirror a subagent's activity
//! into the parent transcript, and that subagent bills its own separate session
//! file. Counting both sides would inflate every total.
//!
//! Assistant lines also carry a `usage` block with the tokens the provider
//! actually billed. That is the ground truth this crate measures against — an
//! estimate can be wrong, a bill cannot.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// What a span of conversation text is, for the purpose of attributing tokens.
///
/// The split matters because compressors target these classes very differently:
/// tool results are bulk data and compress hard, while assistant reasoning is
/// what the model needs to follow its own logic and compresses badly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SegmentKind {
    /// What the human typed.
    UserText,
    /// Assistant prose shown to the user.
    AssistantText,
    /// Extended-thinking blocks.
    Thinking,
    /// The assistant's tool invocation, including serialized arguments.
    ToolUse,
    /// The tool's output, fed back to the model.
    ToolResult,
    /// An image in a tool result. Counted as occurrences, never as text.
    Image,
}

impl SegmentKind {
    pub fn as_str(self) -> &'static str {
        match self {
            SegmentKind::UserText => "user_text",
            SegmentKind::AssistantText => "assistant_text",
            SegmentKind::Thinking => "thinking",
            SegmentKind::ToolUse => "tool_use",
            SegmentKind::ToolResult => "tool_result",
            SegmentKind::Image => "image",
        }
    }

    /// Every kind, in report order.
    pub const ALL: [SegmentKind; 6] = [
        SegmentKind::UserText,
        SegmentKind::AssistantText,
        SegmentKind::Thinking,
        SegmentKind::ToolUse,
        SegmentKind::ToolResult,
        SegmentKind::Image,
    ];
}

/// One attributable span of a transcript.
#[derive(Debug, Clone)]
pub struct Segment {
    pub kind: SegmentKind,
    pub text: String,
    /// Tool name, when the segment belongs to a tool call or its result.
    pub tool: Option<String>,
    /// The call's serialized arguments, carried onto its RESULT too.
    ///
    /// A result on its own cannot say what was asked for, and the question is
    /// what makes two tools comparable: `Read` of a path and an outline of the
    /// same path answer the same request at different cost.
    pub args: Option<String>,
}

/// Tokens the provider billed for one assistant turn.
///
/// Kept separate from any local count. `input_tokens` excludes what was served
/// from cache, so the three input figures are additive, not overlapping.
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct TurnUsage {
    pub input: u64,
    pub cache_creation: u64,
    pub cache_read: u64,
    pub output: u64,
    /// Cache writes at the 5-minute TTL, billed at 1.25x base input.
    pub cache_write_5m: u64,
    /// Cache writes at the 1-hour TTL, billed at 2x base input.
    pub cache_write_1h: u64,
}

impl TurnUsage {
    /// Every input token the request carried, cached or not.
    pub fn input_total(&self) -> u64 {
        self.input + self.cache_creation + self.cache_read
    }

    /// Input tokens re-expressed as multiples of the base input price.
    ///
    /// This is the figure that decides whether a compression pass pays for
    /// itself. A raw token count treats a cache read and a 1-hour cache write
    /// as equal when they differ in price by twenty times.
    pub fn billable_input_units(&self) -> f64 {
        self.input as f64
            + self.cache_read as f64 * 0.1
            + self.cache_write_5m as f64 * 1.25
            + self.cache_write_1h as f64 * 2.0
    }
}

/// One loaded session.
#[derive(Debug, Clone)]
pub struct Session {
    pub path: PathBuf,
    pub segments: Vec<Segment>,
    /// One entry per assistant turn that reported usage.
    pub usage: Vec<TurnUsage>,
    /// Assistant turns seen, whether or not they reported usage.
    pub assistant_turns: usize,
    /// Lines that were not valid JSON. Non-zero means a truncated write.
    pub malformed_lines: usize,
    /// The `message` objects exactly as recorded, in order.
    ///
    /// Kept verbatim rather than rebuilt from `segments`, because an external
    /// compressor routes on message role and block structure. Reconstructing
    /// that from a flattened segment list would benchmark the reconstruction.
    pub raw_messages: Vec<serde_json::Value>,
}

impl Session {
    /// Billed totals across every turn of the session.
    pub fn billed(&self) -> TurnUsage {
        self.usage.iter().fold(TurnUsage::default(), |mut a, u| {
            a.input += u.input;
            a.cache_creation += u.cache_creation;
            a.cache_read += u.cache_read;
            a.output += u.output;
            a.cache_write_5m += u.cache_write_5m;
            a.cache_write_1h += u.cache_write_1h;
            a
        })
    }

    /// Render the session as pipeline input.
    ///
    /// Thinking blocks are dropped: the provider does not re-send them on the
    /// next request, so including them would measure tokens nobody pays twice
    /// for. Images become a short placeholder because they carry no text.
    pub fn to_messages(&self) -> Vec<crate::types::Message> {
        use crate::types::Message;
        let mut out = Vec::new();
        for s in &self.segments {
            let m = match s.kind {
                SegmentKind::Thinking => continue,
                SegmentKind::UserText => Message::user(s.text.clone()),
                SegmentKind::AssistantText => Message::assistant(s.text.clone()),
                SegmentKind::ToolUse => Message::assistant(format!(
                    "[Tool: {}]\n{}",
                    s.tool.as_deref().unwrap_or("unknown"),
                    s.text
                )),
                SegmentKind::ToolResult => Message::tool(s.text.clone()),
                SegmentKind::Image => Message::tool("[image]".to_string()),
            };
            out.push(m);
        }
        out
    }
}

/// Pull the text out of one `tool_result` content field.
///
/// The field is a string in the common case and a block list otherwise, so both
/// shapes are handled rather than assumed.
fn tool_result_segments(
    content: &serde_json::Value,
    tool: Option<String>,
    args: Option<String>,
    out: &mut Vec<Segment>,
) {
    match content {
        serde_json::Value::String(s) => out.push(Segment {
            kind: SegmentKind::ToolResult,
            text: s.clone(),
            tool,
            args,
        }),
        serde_json::Value::Array(items) => {
            for item in items {
                let ty = item.get("type").and_then(|v| v.as_str()).unwrap_or("");
                match ty {
                    "image" => out.push(Segment {
                        kind: SegmentKind::Image,
                        text: String::new(),
                        tool: tool.clone(),
                        args: args.clone(),
                    }),
                    _ => {
                        // `text` and `tool_reference` both carry a text field.
                        if let Some(t) = item.get("text").and_then(|v| v.as_str()) {
                            out.push(Segment {
                                kind: SegmentKind::ToolResult,
                                text: t.to_string(),
                                tool: tool.clone(),
                                args: args.clone(),
                            });
                        }
                    }
                }
            }
        }
        // A null or numeric tool_result carries nothing worth counting.
        _ => {}
    }
}

/// Parse one transcript file.
///
/// A malformed line is counted and skipped, never fatal: these files are
/// appended to by a live process, so the last line of an active session is
/// routinely a partial write.
pub fn load_session(path: &Path) -> std::io::Result<Session> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut segments = Vec::new();
    // tool_use_id -> tool name, so a result can be attributed to its tool.
    // A call always precedes its result in the file, so one forward pass is
    // enough and no second read is needed.
    let mut tool_use_names: std::collections::HashMap<String, (String, String)> =
        std::collections::HashMap::new();
    let mut usage = Vec::new();
    let mut assistant_turns = 0usize;
    let mut malformed_lines = 0usize;
    let mut raw_messages: Vec<serde_json::Value> = Vec::new();

    for line in reader.lines() {
        let line = match line {
            Ok(l) => l,
            // Invalid UTF-8 in one line must not abandon the rest of the file.
            Err(_) => {
                malformed_lines += 1;
                continue;
            }
        };
        if line.trim().is_empty() {
            continue;
        }
        let ev: serde_json::Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(_) => {
                malformed_lines += 1;
                continue;
            }
        };

        let ty = ev.get("type").and_then(|v| v.as_str()).unwrap_or("");
        // Only these two carry conversation content. See the module note on
        // why `progress` is excluded.
        if ty != "user" && ty != "assistant" {
            continue;
        }

        let msg = match ev.get("message") {
            Some(m) if m.is_object() => m,
            _ => continue,
        };
        let role = msg.get("role").and_then(|v| v.as_str()).unwrap_or("");

        // Keep only what the wire format carries; `usage` and harness fields
        // are not part of a request and would distort a token measurement.
        if let Some(c) = msg.get("content") {
            raw_messages.push(serde_json::json!({ "role": role, "content": c }));
        }

        if role == "assistant" {
            assistant_turns += 1;
            if let Some(u) = msg.get("usage") {
                // The TTL split lives in a nested object. When it is absent the
                // whole write is attributed to the 5-minute tier, which is the
                // cheaper of the two — an unknown must not inflate a cost claim.
                let cc = u.get("cache_creation");
                let w5 = cc
                    .and_then(|c| c.get("ephemeral_5m_input_tokens"))
                    .and_then(|v| v.as_u64());
                let w1h = cc
                    .and_then(|c| c.get("ephemeral_1h_input_tokens"))
                    .and_then(|v| v.as_u64());
                let total_write = u
                    .get("cache_creation_input_tokens")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
                let (cache_write_5m, cache_write_1h) = match (w5, w1h) {
                    (Some(a), Some(b)) => (a, b),
                    _ => (total_write, 0),
                };
                usage.push(TurnUsage {
                    input: u.get("input_tokens").and_then(|v| v.as_u64()).unwrap_or(0),
                    cache_creation: total_write,
                    cache_read: u
                        .get("cache_read_input_tokens")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0),
                    output: u.get("output_tokens").and_then(|v| v.as_u64()).unwrap_or(0),
                    cache_write_5m,
                    cache_write_1h,
                });
            }
        }

        match msg.get("content") {
            Some(serde_json::Value::String(s)) => {
                let kind = if role == "assistant" {
                    SegmentKind::AssistantText
                } else {
                    SegmentKind::UserText
                };
                segments.push(Segment {
                    kind,
                    text: s.clone(),
                    tool: None,
                    args: None,
                });
            }
            Some(serde_json::Value::Array(blocks)) => {
                for b in blocks {
                    let bty = b.get("type").and_then(|v| v.as_str()).unwrap_or("");
                    match bty {
                        "text" => {
                            let t = b.get("text").and_then(|v| v.as_str()).unwrap_or("");
                            let kind = if role == "assistant" {
                                SegmentKind::AssistantText
                            } else {
                                SegmentKind::UserText
                            };
                            segments.push(Segment {
                                kind,
                                text: t.to_string(),
                                tool: None,
                                args: None,
                            });
                        }
                        "thinking" => {
                            let t = b.get("thinking").and_then(|v| v.as_str()).unwrap_or("");
                            segments.push(Segment {
                                kind: SegmentKind::Thinking,
                                text: t.to_string(),
                                tool: None,
                                args: None,
                            });
                        }
                        "tool_use" => {
                            let name = b
                                .get("name")
                                .and_then(|v| v.as_str())
                                .unwrap_or("unknown")
                                .to_string();
                            // Arguments are what the request actually carries,
                            // so they are measured as sent, not summarized.
                            let args = b
                                .get("input")
                                .map(|v| v.to_string())
                                .unwrap_or_else(|| "{}".to_string());
                            if let Some(id) = b.get("id").and_then(|v| v.as_str()) {
                                tool_use_names.insert(
                                    id.to_string(),
                                    (name.clone(), args.clone()),
                                );
                            }
                            segments.push(Segment {
                                kind: SegmentKind::ToolUse,
                                text: args.clone(),
                                tool: Some(name),
                                args: Some(args),
                            });
                        }
                        "tool_result" => {
                            // A result names only the id of the call it answers.
                            // Resolving it back to the tool name is what makes
                            // "which tool's OUTPUT costs the most" answerable —
                            // and output dwarfs arguments, so the unresolved
                            // form attributes the small half and drops the big.
                            let found = b
                                .get("tool_use_id")
                                .and_then(|v| v.as_str())
                                .and_then(|id| tool_use_names.get(id).cloned());
                            let (tool, args) = match found {
                                Some((n, a)) => (Some(n), Some(a)),
                                None => (None, None),
                            };
                            if let Some(c) = b.get("content") {
                                tool_result_segments(c, tool, args, &mut segments);
                            }
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }

    Ok(Session {
        path: path.to_path_buf(),
        segments,
        usage,
        assistant_turns,
        malformed_lines,
        raw_messages,
    })
}

/// Recursively collect every `.jsonl` transcript under `root`.
pub fn find_transcripts(root: &Path) -> std::io::Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let entries = match std::fs::read_dir(&dir) {
            Ok(e) => e,
            // An unreadable directory is skipped, not fatal: the corpus is
            // large and one bad permission must not lose the other 13,000.
            Err(_) => continue,
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().and_then(|e| e.to_str()) == Some("jsonl") {
                out.push(path);
            }
        }
    }
    out.sort();
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_tmp(name: &str, body: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("distil-corpus-test-{name}.jsonl"));
        let mut f = File::create(&p).unwrap();
        f.write_all(body.as_bytes()).unwrap();
        p
    }

    #[test]
    fn parses_roles_and_block_kinds() {
        let body = r#"{"type":"user","message":{"role":"user","content":"fix the build"}}
{"type":"assistant","message":{"role":"assistant","content":[{"type":"thinking","thinking":"consider"},{"type":"text","text":"Looking."},{"type":"tool_use","name":"Bash","input":{"command":"cargo build"}}],"usage":{"input_tokens":10,"cache_creation_input_tokens":5,"cache_read_input_tokens":100,"output_tokens":7}}}
{"type":"user","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"error[E0308]"}]}}
"#;
        let p = write_tmp("basic", body);
        let s = load_session(&p).unwrap();

        let kinds: Vec<_> = s.segments.iter().map(|x| x.kind).collect();
        assert_eq!(
            kinds,
            vec![
                SegmentKind::UserText,
                SegmentKind::Thinking,
                SegmentKind::AssistantText,
                SegmentKind::ToolUse,
                SegmentKind::ToolResult,
            ]
        );
        assert_eq!(s.assistant_turns, 1);
        assert_eq!(s.billed().output, 7);
        assert_eq!(s.billed().input_total(), 115);
        // No TTL breakdown present, so the write lands on the cheaper tier.
        assert_eq!(s.billed().cache_write_5m, 5);
        assert_eq!(s.billed().cache_write_1h, 0);
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn skips_progress_and_bookkeeping_lines() {
        // A progress line mirrors a subagent that bills its own session file.
        // Counting it here would double-count those tokens.
        let body = r#"{"type":"progress","message":{"role":"assistant","content":[{"type":"text","text":"subagent chatter"}]}}
{"type":"attachment","message":{"role":"user","content":"pasted blob"}}
{"type":"file-history-snapshot"}
{"type":"user","message":{"role":"user","content":"real prompt"}}
"#;
        let p = write_tmp("skip", body);
        let s = load_session(&p).unwrap();
        assert_eq!(s.segments.len(), 1);
        assert_eq!(s.segments[0].text, "real prompt");
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn tool_result_block_list_and_images() {
        let body = r#"{"type":"user","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":[{"type":"text","text":"line one"},{"type":"image","source":{}},{"type":"tool_reference","text":"ref"}]}]}}
"#;
        let p = write_tmp("blocks", body);
        let s = load_session(&p).unwrap();
        let kinds: Vec<_> = s.segments.iter().map(|x| x.kind).collect();
        assert_eq!(
            kinds,
            vec![
                SegmentKind::ToolResult,
                SegmentKind::Image,
                SegmentKind::ToolResult
            ]
        );
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn malformed_lines_are_counted_not_fatal() {
        let body = "{\"type\":\"user\",\"message\":{\"role\":\"user\",\"content\":\"ok\"}}\n{ this is not json\n";
        let p = write_tmp("malformed", body);
        let s = load_session(&p).unwrap();
        assert_eq!(s.malformed_lines, 1);
        assert_eq!(s.segments.len(), 1);
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn thinking_is_dropped_from_pipeline_input() {
        // The provider does not resend thinking blocks, so they are not part of
        // the next request's input and must not be measured as if they were.
        let body = r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"thinking","thinking":"long private reasoning"},{"type":"text","text":"short answer"}]}}
"#;
        let p = write_tmp("thinking", body);
        let s = load_session(&p).unwrap();
        let msgs = s.to_messages();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].content, "short answer");
        std::fs::remove_file(&p).ok();
    }
}

#[cfg(test)]
mod pricing_tests {
    use super::*;

    #[test]
    fn ttl_split_is_read_when_present() {
        use std::io::Write;
        let body = r#"{"type":"assistant","message":{"role":"assistant","content":[],"usage":{"input_tokens":0,"cache_creation_input_tokens":300,"cache_read_input_tokens":0,"output_tokens":0,"cache_creation":{"ephemeral_5m_input_tokens":100,"ephemeral_1h_input_tokens":200}}}}
"#;
        let mut p = std::env::temp_dir();
        p.push("distil-corpus-test-ttl.jsonl");
        std::fs::File::create(&p)
            .unwrap()
            .write_all(body.as_bytes())
            .unwrap();
        let s = load_session(&p).unwrap();
        let b = s.billed();
        assert_eq!(b.cache_write_5m, 100);
        assert_eq!(b.cache_write_1h, 200);
        // 100 * 1.25 + 200 * 2.0 = 525
        assert!((b.billable_input_units() - 525.0).abs() < 1e-9);
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn cache_read_is_a_tenth_of_base_price() {
        let u = TurnUsage {
            input: 0,
            cache_creation: 0,
            cache_read: 1_000,
            output: 0,
            cache_write_5m: 0,
            cache_write_1h: 0,
        };
        assert!((u.billable_input_units() - 100.0).abs() < 1e-9);
    }
}

#[cfg(test)]
mod attribution_tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn tool_results_resolve_back_to_their_tool_name() {
        // Output is far larger than arguments, so a result that cannot name its
        // tool leaves the expensive half of the corpus unattributed.
        let body = r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","id":"tu_1","name":"Read","input":{"file":"a.rs"}}]}}
{"type":"user","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"tu_1","content":"fn main() {}"}]}}
{"type":"user","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"tu_missing","content":"orphan"}]}}
"#;
        let mut p = std::env::temp_dir();
        p.push("distil-corpus-test-attrib.jsonl");
        File::create(&p).unwrap().write_all(body.as_bytes()).unwrap();
        let s = load_session(&p).unwrap();

        let results: Vec<_> = s
            .segments
            .iter()
            .filter(|x| x.kind == SegmentKind::ToolResult)
            .collect();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].tool.as_deref(), Some("Read"));
        // An unmatched id stays None rather than being invented.
        assert_eq!(results[1].tool, None);
        std::fs::remove_file(&p).ok();
    }
}
