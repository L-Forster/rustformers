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
    
    // Create random tensors
    let q: Tensor<TestBackend, 3> = Tensor::random([batch_size, seq_len, d_model], Distribution::Standard, &device);
    let k: Tensor<TestBackend, 3> = Tensor::random([batch_size, seq_len, d_model], Distribution::Standard, &device);
    let v: Tensor<TestBackend, 3> = Tensor::random([batch_size, seq_len, d_model], Distribution::Standard, &device);

    let output = mha(q.clone(), k, v);
    
    assert_eq!(output.dims(), [batch_size, seq_len, d_model]);
}

#[test]
fn test_mha_attention_mechanism() {
    let device = Default::default();
    
    // Simple case: 1 batch, 2 tokens, embedding dim 2
    // Q matches K exactly for the first position to attend to first position strongly
    // But softmax will distribute it.
    
    // Q = [[100.0, 0.0], [0.0, 100.0]]
    // K = [[100.0, 0.0], [0.0, 100.0]]
    // V = [[1.0, 2.0], [3.0, 4.0]]
    
    // Score[0,0] = (100*100 + 0) / sqrt(2) = 10000/1.414 = 7072
    // Score[0,1] = 0
    // Softmax([7072, 0]) -> [1.0, 0.0] (approx)
    // Output[0] = 1.0 * V[0] + 0.0 * V[1] = [1.0, 2.0]
    
    let q_data = [100.0, 0.0, 0.0, 100.0];
    let k_data = [100.0, 0.0, 0.0, 100.0];
    let v_data = [1.0, 2.0, 3.0, 4.0];
    
    let q: Tensor<TestBackend, 3> = Tensor::from_floats(q_data, &device).reshape([1, 2, 2]);
    let k: Tensor<TestBackend, 3> = Tensor::from_floats(k_data, &device).reshape([1, 2, 2]);
    let v: Tensor<TestBackend, 3> = Tensor::from_floats(v_data, &device).reshape([1, 2, 2]);
    
    let output = mha(q, k, v);
    let output_data = output.into_data();
    
    // Check values
    // We expect output to be close to V because Q and K are aligned diagonal matrices with large values
    // so attention matrix is Identity.
    
    let vals = output_data.convert::<f32>().value;
    
    // vals should be [1.0, 2.0, 3.0, 4.0]
    assert!((vals[0] - 1.0).abs() < 1e-3);
    assert!((vals[1] - 2.0).abs() < 1e-3);
    assert!((vals[2] - 3.0).abs() < 1e-3);
    assert!((vals[3] - 4.0).abs() < 1e-3);
}

