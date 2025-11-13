// Byte-Pair Embedding implementation. 
// To be implemented.
use std::io::{self, Write};
use std::fs;

fn load_corpus() -> Vec<String> {
    let temp = fs::read_to_string("data/corpus.txt")
        .expect("Failed to read corpus file")
        .lines()
        .map(|line| line.to_string())
        .collect();
    println!("Corpus loaded successfully.");
    temp
}

fn main() {
    println!("Byte-Pair Embedding module loaded.");
    let corpus: Vec<String> = load_corpus();
    println!("Loaded corpus with {} lines", corpus.len());
    println!("{:?}", corpus);
}  
