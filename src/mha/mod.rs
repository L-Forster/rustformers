// Multi-Head self-attention implementation. 
use burn::tensor::{backend::Backend, Tensor};

/// Multi-head attention (or rather, scaled dot-product attention) implementation.
/// 
/// Expects inputs to be already projected and reshaped if multi-head is desired.
/// Shapes should be [..., seq_len, d_k]
pub fn mha<B: Backend, const D: usize>(
    q: Tensor<B, D>, 
    k: Tensor<B, D>, 
    v: Tensor<B, D>
) -> Tensor<B, D> {
    let dim_k = k.dims()[D - 1] as f64;
    
    // Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
    
    // Swap last two dimensions of K to transpose it for matmul
    // K: [..., seq, d] -> K^T: [..., d, seq]
    let k_t = k.swap_dims(D - 2, D - 1);
    
    // Q * K^T -> [..., seq, seq] (scores)
    let scores = q.matmul(k_t) / dim_k.sqrt();
    
    // Softmax over the last dimension (key sequence dimension)
    let weights = scores.softmax(D - 1);
    
    // Weights * V -> [..., seq, d]
    weights.matmul(v)
}

#[cfg(test)]
mod tests;

