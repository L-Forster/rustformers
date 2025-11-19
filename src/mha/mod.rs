use burn::{
    nn::{Linear, LinearConfig, Dropout, DropoutConfig},
    prelude::*,
    tensor::activation::softmax,
};

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
        
        let query = LinearConfig::new(self.d_model, self.d_model)
            .with_bias(self.bias)
            .init(device);
        let key = LinearConfig::new(self.d_model, self.d_model)
            .with_bias(self.bias)
            .init(device);
        let value = LinearConfig::new(self.d_model, self.d_model)
            .with_bias(self.bias)
            .init(device);
        let output = LinearConfig::new(self.d_model, self.d_model)
            .with_bias(self.bias)
            .init(device);
        let dropout = DropoutConfig::new(self.dropout).init();

        MultiHeadAttention {
            query,
            key,
            value,
            output,
            dropout,
            n_heads: self.n_heads,
            d_head,
            d_model: self.d_model,
        }
    }
}

#[derive(Module, Debug)]
pub struct MultiHeadAttention<B: Backend> {
    query: Linear<B>,
    key: Linear<B>,
    value: Linear<B>,
    output: Linear<B>,
    dropout: Dropout,
    n_heads: usize,
    d_head: usize,
    d_model: usize,
}

impl<B: Backend> MultiHeadAttention<B> {
    pub fn compute_multi_head_attention(
        &self, 
        q: Tensor<B, 3>, 
        k: Tensor<B, 3>, 
        v: Tensor<B, 3>, 
        mask: Option<Tensor<B, 4>>
    ) -> Tensor<B, 3> {
        let [batch_size, seq_len, _] = q.dims();

        // 1. Linear Projections
        let q = self.query.forward(q);
        let k = self.key.forward(k);
        let v = self.value.forward(v);

        // 2. Split Heads & Transpose
        // [batch, seq, d_model] -> [batch, seq, heads, d_head]
        let q = q.reshape([batch_size, seq_len, self.n_heads, self.d_head]);
        let k = k.reshape([batch_size, seq_len, self.n_heads, self.d_head]);
        let v = v.reshape([batch_size, seq_len, self.n_heads, self.d_head]);

        // [batch, seq, heads, d_head] -> [batch, heads, seq, d_head]
        let q = q.swap_dims(1, 2);
        let k = k.swap_dims(1, 2);
        let v = v.swap_dims(1, 2);

        // 3. Scaled Dot-Product Attention
        let output = scaled_dot_product_attention(q, k, v, mask, &self.dropout);

        // 4. Merge Heads
        // [batch, heads, seq, d_head] -> [batch, seq, heads, d_head]
        let output = output.swap_dims(1, 2);
        
        // [batch, seq, heads, d_head] -> [batch, seq, d_model]
        let output = output.reshape([batch_size, seq_len, self.d_model]);

        // 5. Output Projection
        self.output.forward(output)
    }
}

fn scaled_dot_product_attention<B: Backend>(
    q: Tensor<B, 4>, 
    k: Tensor<B, 4>, 
    v: Tensor<B, 4>, 
    mask: Option<Tensor<B, 4>>,
    dropout: &Dropout
) -> Tensor<B, 4> {
    let d_k = k.dims()[3] as f64;
    
    // Q * K^T -> [batch, heads, seq, seq]
    // Swap last two dimensions of K to transpose it for matmul
    let k_t = k.swap_dims(2, 3);
    let mut scores = q.matmul(k_t) / d_k.sqrt();
    
    if let Some(mask) = mask {
        scores = scores + mask;
    }

    // Softmax over the last dimension
    let weights = softmax(scores, 3);
    
    // Apply dropout to attention weights
    let weights = dropout.forward(weights);
    
    // Weights * V -> [batch, heads, seq, d_head]
    weights.matmul(v)
}

#[cfg(test)]
mod tests;
