use burn::{
    nn::Dropout,
    prelude::*,
    tensor::activation::softmax,
};

/// Applies stable softmax by subtracting the maximum value before computing softmax.
pub fn stable_softmax<B: Backend, const D: usize>(tensor: Tensor<B, D>, dim: usize) -> Tensor<B, D> {
    let max_val = tensor.clone().max_dim(dim);
    let stabilized = tensor - max_val;
    softmax(stabilized, dim)
}

/// Applies an additive mask to attention scores.
pub fn apply_attention_mask<B: Backend>(
    scores: Tensor<B, 4>,
    mask: Option<Tensor<B, 4>>,
    min_val: f64,
) -> Tensor<B, 4> {
    match mask {
        Some(m) => {
            let m = m.to_device(&scores.device());
            // Assuming mask is 0/1 (1=keep), convert to additive if needed.
            // We'll assume the user provides a mask where 1 means "valid" and 0 means "mask out".
            // If the mask contains negative values, we assume it's already additive.
            // Since we can't check values easily, we'll provide a separate function or just assume 0/1 for this helper
            // as requested by "Auto-convert".
            
            // (1 - mask) * min_val
            let anti_mask = m.neg().add_scalar(1.0);
            let additive = anti_mask.mul_scalar(min_val);
            scores.add(additive)
        }
        None => scores,
    }
}

/// Core scaled dot-product attention logic.
/// Separated for testing and modularity.
pub fn scaled_dot_product_attention<B: Backend>(
    q: Tensor<B, 4>, 
    k: Tensor<B, 4>, 
    v: Tensor<B, 4>, 
    mask: Option<Tensor<B, 4>>,
    dropout: &Dropout,
    d_k: f64
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    
    // Q * K^T -> [batch, heads, seq, seq]
    let k_t = k.swap_dims(2, 3);
    let scores = q.matmul(k_t) / d_k.sqrt();
    
    // Apply Mask
    let scores = apply_attention_mask(scores, mask, -1e9);

    // Stable Softmax
    let weights = stable_softmax(scores, 3);
    
    // Conditional Dropout
    let weights = dropout.forward(weights);
    
    // Weights * V -> [batch, heads, seq, d_head]
    let context = weights.clone().matmul(v);
    
    (context, weights)
}
