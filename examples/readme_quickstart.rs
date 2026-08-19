//! The example printed in README.md, kept compilable.
//!
//! A README snippet that does not build is worse than none: it is the first
//! thing a reader tries. This file is the same code, so `cargo build
//! --examples` fails the moment the two drift apart.

use distil::{CacheAlignLayer, Ctx, EstimateCounter, Message, Pipeline, ToolSpec};

fn main() {
    let messages: Vec<Message> = vec![
        Message::system("You are a coding agent."),
        Message::user("Fix the failing build."),
    ];
    let tools: Vec<ToolSpec> = vec![];
    let turn = 1;

    let pipeline = Pipeline::builder()
        .counter(EstimateCounter)
        .layer(CacheAlignLayer::generic())
        .build();

    let mut ctx = Ctx::new(messages, tools, turn);
    let result = pipeline.optimize(&mut ctx);
    println!("{result}");
}
