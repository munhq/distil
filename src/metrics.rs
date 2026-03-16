//! Prometheus metrics for distil pipeline observability.
//!
//! Provides [`DistilMetrics`] — a standalone observer that records metrics from
//! [`PipelineResult`]. Feature-gated behind `metrics`.
//!
//! # Usage
//!
//! ```rust,ignore
//! let metrics = DistilMetrics::new();
//!
//! // After each optimization:
//! let result = pipeline.optimize(&mut ctx);
//! metrics.record(&result);
//!
//! // Expose via HTTP:
//! let output = metrics.render();  // Prometheus text format
//! ```
//!
//! # Metrics Exported
//!
//! | Metric | Type | Labels | Description |
//! |--------|------|--------|-------------|
//! | `distil_tokens_saved_total` | Counter | `layer` | Cumulative tokens saved |
//! | `distil_tokens_before_total` | Counter | — | Cumulative input tokens |
//! | `distil_tokens_after_total` | Counter | — | Cumulative output tokens |
//! | `distil_layer_duration_seconds` | Histogram | `layer` | Per-layer latency |
//! | `distil_pipeline_duration_seconds` | Histogram | — | Total pipeline latency |
//! | `distil_compression_ratio` | Gauge | — | Last compression ratio (0.0–1.0) |
//! | `distil_optimizations_total` | Counter | — | Total optimize() calls |

use prometheus::{
    Encoder, GaugeVec, HistogramOpts, HistogramVec, IntCounter, IntCounterVec, Opts, Registry,
    TextEncoder,
};

use crate::pipeline::PipelineResult;

/// Prometheus metrics collector for distil pipeline operations.
///
/// Thread-safe — all prometheus types use internal `Arc` synchronization.
/// Create one instance at startup and share it (e.g., via `Arc<DistilMetrics>`
/// in axum state).
pub struct DistilMetrics {
    registry: Registry,
    tokens_saved: IntCounterVec,
    tokens_before: IntCounter,
    tokens_after: IntCounter,
    layer_duration: HistogramVec,
    pipeline_duration: prometheus::Histogram,
    compression_ratio: GaugeVec,
    optimizations: IntCounter,
    probe_success: IntCounterVec,
    probe_failure: IntCounterVec,
    probe_success_rate: GaugeVec,
}

impl DistilMetrics {
    /// Create a new metrics collector with a fresh Prometheus registry.
    pub fn new() -> Self {
        let registry = Registry::new_custom(Some("distil".into()), None)
            .expect("failed to create prometheus registry");

        let tokens_saved = IntCounterVec::new(
            Opts::new("tokens_saved_total", "Cumulative tokens saved by layer"),
            &["layer"],
        )
        .expect("metric creation failed");

        let tokens_before = IntCounter::new(
            "tokens_before_total",
            "Cumulative input tokens before optimization",
        )
        .expect("metric creation failed");

        let tokens_after = IntCounter::new(
            "tokens_after_total",
            "Cumulative output tokens after optimization",
        )
        .expect("metric creation failed");

        let layer_duration = HistogramVec::new(
            HistogramOpts::new(
                "layer_duration_seconds",
                "Per-layer optimization latency in seconds",
            )
            .buckets(vec![
                0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0,
            ]),
            &["layer"],
        )
        .expect("metric creation failed");

        let pipeline_duration = prometheus::Histogram::with_opts(
            HistogramOpts::new(
                "pipeline_duration_seconds",
                "Total pipeline optimization latency in seconds",
            )
            .buckets(vec![
                0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0,
            ]),
        )
        .expect("metric creation failed");

        let compression_ratio = GaugeVec::new(
            Opts::new(
                "compression_ratio",
                "Last compression ratio (tokens_after / tokens_before, lower is better)",
            ),
            &["scope"],
        )
        .expect("metric creation failed");

        let optimizations = IntCounter::new(
            "optimizations_total",
            "Total number of optimize() calls",
        )
        .expect("metric creation failed");

        let probe_success = IntCounterVec::new(
            Opts::new("probe_success_total", "Probe questions answered correctly"),
            &["type"],
        )
        .expect("metric creation failed");

        let probe_failure = IntCounterVec::new(
            Opts::new("probe_failure_total", "Probe questions answered incorrectly"),
            &["type"],
        )
        .expect("metric creation failed");

        let probe_success_rate = GaugeVec::new(
            Opts::new(
                "probe_success_rate",
                "Latest probe success rate (0.0-1.0)",
            ),
            &["scope"],
        )
        .expect("metric creation failed");

        // Register all metrics
        registry
            .register(Box::new(tokens_saved.clone()))
            .expect("register failed");
        registry
            .register(Box::new(tokens_before.clone()))
            .expect("register failed");
        registry
            .register(Box::new(tokens_after.clone()))
            .expect("register failed");
        registry
            .register(Box::new(layer_duration.clone()))
            .expect("register failed");
        registry
            .register(Box::new(pipeline_duration.clone()))
            .expect("register failed");
        registry
            .register(Box::new(compression_ratio.clone()))
            .expect("register failed");
        registry
            .register(Box::new(optimizations.clone()))
            .expect("register failed");
        registry
            .register(Box::new(probe_success.clone()))
            .expect("register failed");
        registry
            .register(Box::new(probe_failure.clone()))
            .expect("register failed");
        registry
            .register(Box::new(probe_success_rate.clone()))
            .expect("register failed");

        Self {
            registry,
            tokens_saved,
            tokens_before,
            tokens_after,
            layer_duration,
            pipeline_duration,
            compression_ratio,
            optimizations,
            probe_success,
            probe_failure,
            probe_success_rate,
        }
    }

    /// Record metrics from a completed pipeline optimization.
    pub fn record(&self, result: &PipelineResult) {
        self.optimizations.inc();
        self.tokens_before.inc_by(result.tokens_before as u64);
        self.tokens_after.inc_by(result.tokens_after as u64);
        self.pipeline_duration
            .observe(result.duration.as_secs_f64());

        // Per-layer metrics
        for lr in &result.layers {
            let saved = lr.tokens_saved() as u64;
            if saved > 0 {
                self.tokens_saved
                    .with_label_values(&[&lr.layer])
                    .inc_by(saved);
            }
            self.layer_duration
                .with_label_values(&[&lr.layer])
                .observe(lr.duration.as_secs_f64());
        }

        // Compression ratio (0.0 = perfect compression, 1.0 = no savings)
        if result.tokens_before > 0 {
            let ratio = result.tokens_after as f64 / result.tokens_before as f64;
            self.compression_ratio
                .with_label_values(&["pipeline"])
                .set(ratio);
        }
    }

    /// Record metrics from a probe evaluation report.
    pub fn record_probes(&self, report: &crate::probe::ProbeReport) {
        for result in &report.results {
            let type_str = result.probe.probe_type.to_string();
            if result.passed {
                self.probe_success.with_label_values(&[&type_str]).inc();
            } else {
                self.probe_failure.with_label_values(&[&type_str]).inc();
            }
        }
        self.probe_success_rate
            .with_label_values(&["pipeline"])
            .set(report.success_rate);
    }

    /// Render all metrics in Prometheus text exposition format.
    pub fn render(&self) -> String {
        let metric_families = self.registry.gather();
        let encoder = TextEncoder::new();
        let mut buffer = Vec::new();
        encoder
            .encode(&metric_families, &mut buffer)
            .expect("prometheus encoding failed");
        String::from_utf8(buffer).expect("prometheus output is not valid UTF-8")
    }

    /// Access the underlying Prometheus registry (for custom metrics).
    pub fn registry(&self) -> &Registry {
        &self.registry
    }
}

impl Default for DistilMetrics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::{LayerResult, PipelineResult};
    use std::time::Duration;

    fn mock_result() -> PipelineResult {
        PipelineResult {
            layers: vec![
                LayerResult {
                    layer: "registry".into(),
                    tokens_before: 5000,
                    tokens_after: 3500,
                    duration: Duration::from_micros(150),
                    detail: "30 tools compressed".into(),
                },
                LayerResult {
                    layer: "masking".into(),
                    tokens_before: 3500,
                    tokens_after: 2000,
                    duration: Duration::from_micros(80),
                    detail: "old results masked".into(),
                },
                LayerResult {
                    layer: "compactor".into(),
                    tokens_before: 2000,
                    tokens_after: 1990,
                    duration: Duration::from_micros(20),
                    detail: "whitespace stripped".into(),
                },
            ],
            tokens_before: 5000,
            tokens_after: 1990,
            duration: Duration::from_micros(300),
        }
    }

    #[test]
    fn records_and_renders_metrics() {
        let metrics = DistilMetrics::new();
        metrics.record(&mock_result());

        let output = metrics.render();

        // Verify key metrics appear in output
        assert!(
            output.contains("distil_tokens_saved_total"),
            "missing tokens_saved_total: {output}"
        );
        assert!(
            output.contains("distil_tokens_before_total"),
            "missing tokens_before_total"
        );
        assert!(
            output.contains("distil_tokens_after_total"),
            "missing tokens_after_total"
        );
        assert!(
            output.contains("distil_layer_duration_seconds"),
            "missing layer_duration"
        );
        assert!(
            output.contains("distil_pipeline_duration_seconds"),
            "missing pipeline_duration"
        );
        assert!(
            output.contains("distil_compression_ratio"),
            "missing compression_ratio"
        );
        assert!(
            output.contains("distil_optimizations_total"),
            "missing optimizations_total"
        );

        // Verify layer labels
        assert!(
            output.contains(r#"layer="registry""#),
            "missing registry label"
        );
        assert!(
            output.contains(r#"layer="masking""#),
            "missing masking label"
        );
    }

    #[test]
    fn accumulates_across_multiple_records() {
        let metrics = DistilMetrics::new();
        let result = mock_result();

        metrics.record(&result);
        metrics.record(&result);
        metrics.record(&result);

        let output = metrics.render();
        // optimizations_total should be 3
        assert!(
            output.contains("distil_optimizations_total 3"),
            "expected 3 optimizations in output: {output}"
        );
    }

    #[test]
    fn records_probe_metrics() {
        use crate::probe::{Probe, ProbeReport, ProbeResult, ProbeType};

        let metrics = DistilMetrics::new();
        let report = ProbeReport {
            results: vec![
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
                        probe_type: ProbeType::Artifact,
                        question: "q2".into(),
                        expected: "a2".into(),
                    },
                    passed: false,
                    actual_answer: "wrong".into(),
                },
            ],
            success_rate: 0.5,
            by_type: {
                let mut m = std::collections::HashMap::new();
                m.insert(ProbeType::Recall, 1.0);
                m.insert(ProbeType::Artifact, 0.0);
                m
            },
        };

        metrics.record_probes(&report);
        let output = metrics.render();

        assert!(
            output.contains("distil_probe_success_total"),
            "missing probe_success_total: {output}"
        );
        assert!(
            output.contains("distil_probe_failure_total"),
            "missing probe_failure_total: {output}"
        );
        assert!(
            output.contains("distil_probe_success_rate"),
            "missing probe_success_rate: {output}"
        );
    }

    #[test]
    fn handles_zero_token_result() {
        let metrics = DistilMetrics::new();
        let result = PipelineResult {
            layers: vec![],
            tokens_before: 0,
            tokens_after: 0,
            duration: Duration::ZERO,
        };
        metrics.record(&result);

        let output = metrics.render();
        assert!(output.contains("distil_optimizations_total 1"));
    }
}
