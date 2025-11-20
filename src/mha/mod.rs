use burn::{
    nn::{Linear, LinearConfig, Dropout, DropoutConfig},
    prelude::*,
};

mod utils;
// Re-export utils if needed, or just use them internally.
// Tests might need access to utils, so we might want to make utils public mod or re-export.
// But usually `mod utils` makes it private to `mha`. `tests` is a child module so it can access `super::utils`.
use utils::{scaled_dot_product_attention};

#[derive(Config, Debug)]
pub struct MhaConfig {
    pub d_model: usize,
    pub n_heads: usize,
    #[config(default = 0.1)]
    pub dropout: f64,
    #[config(default = true)]
    pub bias: bool,
}

impl MhaConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> MultiHeadAttention<B> {
        let d_head = self.d_model / self.n_heads;
        
        // Fused QKV projection: [d_model] -> [3 * d_model]
        let qkv = LinearConfig::new(self.d_model, self.d_model * 3)
            .with_bias(self.bias)
            .init(device);
            
        let output = LinearConfig::new(self.d_model, self.d_model)
            .with_bias(self.bias)
            .init(device);
            
        let dropout = DropoutConfig::new(self.dropout).init();

        MultiHeadAttention {
            qkv,
            output,
            dropout,
            n_heads: self.n_heads,
            d_head,
            d_model: self.d_model,
        }
    }
}

pub struct MhaOutput<B: Backend> {
    pub context: Tensor<B, 3>,
    pub weights: Option<Tensor<B, 4>>,
}

#[derive(Module, Debug)]
pub struct MultiHeadAttention<B: Backend> {
    qkv: Linear<B>,
    output: Linear<B>,
    dropout: Dropout,
    n_heads: usize,
    d_head: usize,
    d_model: usize,
}

impl<B: Backend> MultiHeadAttention<B> {
    /// Forward pass for Multi-Head Self-Attention.
    pub fn forward(
        &self, 
        input: Tensor<B, 3>, 
        mask: Option<Tensor<B, 4>>
    ) -> MhaOutput<B> {
        let [batch_size, seq_len, d_model_in] = input.dims();
        
        debug_assert_eq!(d_model_in, self.d_model, "Input embedding dim must match config");

        // 1. Fused QKV
        let qkv_out = self.qkv.forward(input);
        
        // 2. Split & Reshape
        let chunks = qkv_out.chunk(3, 2);
        let (q, k, v) = (chunks[0].clone(), chunks[1].clone(), chunks[2].clone());
        
        let q = q.reshape([batch_size, seq_len, self.n_heads, self.d_head]).swap_dims(1, 2);
        let k = k.reshape([batch_size, seq_len, self.n_heads, self.d_head]).swap_dims(1, 2);
        let v = v.reshape([batch_size, seq_len, self.n_heads, self.d_head]).swap_dims(1, 2);
        
        #[cfg(debug_assertions)]
        {
             let [b, h, s, d] = q.dims();
             debug_assert_eq!(b, batch_size);
             debug_assert_eq!(h, self.n_heads);
             debug_assert_eq!(s, seq_len);
             debug_assert_eq!(d, self.d_head);
        }

        // 3. Scaled Dot-Product Attention
        let (context, weights) = scaled_dot_product_attention(
            q, k, v, mask, &self.dropout, self.d_head as f64
        );

        // 4. Merge Heads
        let context = context.swap_dims(1, 2).reshape([batch_size, seq_len, self.d_model]);

        // 5. Output Projection
        let context = self.output.forward(context);
        
        MhaOutput {
            context,
            weights: Some(weights),
        }
    }
}

#[cfg(test)]
mod tests;
