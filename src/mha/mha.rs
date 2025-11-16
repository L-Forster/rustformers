// Multi-Head self-attention implementation. 
// To be implemented.

use burn::{
    tensor::{backend::Backend, Tensor},
    Data,
};

fn mha(q: Tensor, k: Tensor, v: Tensor) -> Tensor {
    let q = q.transpose(-2, -1);
    let k = k.transpose(-2, -1);
    let v = v.transpose(-2, -1);
    let qk = q.matmul(k).softmax(-1);
    let qk = qk.matmul(v);
    qk.transpose(-2, -1)
}

fn main() {
    println!("Multi-Head self-attention module loaded.");
}
