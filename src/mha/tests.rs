use super::*;
use burn::tensor::{Tensor, Distribution};
use burn_ndarray::NdArray;

type TestBackend = NdArray<f32>;

#[test]
fn test_mha_shape() {
    let device = Default::default();
    let batch_size = 2;
    let seq_len = 4;
    let d_model = 8;
    let n_heads = 2;
    
    let config = MhaConfig::new(d_model, n_heads);
    let mha = config.init::<TestBackend>(&device);

    // Create random tensors
    let q: Tensor<TestBackend, 3> = Tensor::random([batch_size, seq_len, d_model], Distribution::Default, &device);
    let k: Tensor<TestBackend, 3> = Tensor::random([batch_size, seq_len, d_model], Distribution::Default, &device);
    let v: Tensor<TestBackend, 3> = Tensor::random([batch_size, seq_len, d_model], Distribution::Default, &device);

    let output = mha.forward(q, k, v);
    
    assert_eq!(output.dims(), [batch_size, seq_len, d_model]);
}

#[test]
fn test_mha_forward() {
    let device = Default::default();
    let d_model = 8;
    let n_heads = 2;
    
    let config = MhaConfig::new(d_model, n_heads);
    let mha = config.init::<TestBackend>(&device);

    let x: Tensor<TestBackend, 3> = Tensor::random([1, 4, d_model], Distribution::Default, &device);
    
    // Self-attention: q=k=v=x
    let output = mha.forward(x.clone(), x.clone(), x.clone());
    
    assert_eq!(output.dims(), [1, 4, d_model]);
}

