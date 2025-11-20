use burn::{prelude::*, tensor::Distribution};
use rustformers::mha::MultiHeadAttention;

fn main() {
    let mha = MultiHeadAttention::new(512, 8);

    let input = Tensor::random([4, 128, 512], Distribution::Default, mha.device());
    println!("Input:  {:?}", input.dims());

    let output = mha.forward(input, None);

    println!("Output: {:?}", output.context.dims());
}
