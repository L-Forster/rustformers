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

    let output = mha.compute_multi_head_attention(q, k, v);
    
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
    let output = mha.compute_multi_head_attention(x.clone(), x.clone(), x.clone());
    
    assert_eq!(output.dims(), [1, 4, d_model]);
}

#[test]
fn test_scaled_dot_product_attention_math() {
    let device = Default::default();
    
    // 1. Setup Inputs
    // Batch=1, Heads=1, Seq=2, Dim=2
    let q_data: [f32; 4] = [1.0, 0.0, 0.0, 1.0]; // Identity
    let k_data: [f32; 4] = [1.0, 0.0, 0.0, 1.0]; // Identity
    // Use denser values to verify mixing across dimensions
    // V1 = [10, 5], V2 = [2, 20]
    let v_data: [f32; 4] = [10.0, 5.0, 2.0, 20.0]; 
    
    let q = Tensor::<TestBackend, 1>::from_floats(q_data, &device).reshape([1, 1, 2, 2]);
    let k = Tensor::<TestBackend, 1>::from_floats(k_data, &device).reshape([1, 1, 2, 2]);
    let v = Tensor::<TestBackend, 1>::from_floats(v_data, &device).reshape([1, 1, 2, 2]);
    
    // 2. Setup Dropout (Disabled)
    let dropout_config = burn::nn::DropoutConfig::new(0.0);
    let dropout = dropout_config.init();

    // 3. Run Function
    // We can access this because tests is a child module of mha
    let output = super::scaled_dot_product_attention(q, k, v, &dropout);
    
    // 4. Manual Calculation Verification
    // d_k = 2.0. sqrt(d_k) = 1.41421356
    // Q @ K^T = [[1, 0], [0, 1]] (Identity matches Identity)
    // Scaled = [[0.7071, 0.0], [0.0, 0.7071]]
    
    // Softmax([0.7071, 0.0]):
    // e^0.7071 = 2.0281
    // e^0 = 1.0
    // Sum = 3.0281
    // p1 = 2.0281 / 3.0281 = 0.6697
    // p2 = 1.0 / 3.0281 = 0.3302
    
    // Output = Weights @ V
    // Row 1 Weights: [0.6697, 0.3302]
    // Row 1 Output:
    // Dim 0: 0.6697*10 + 0.3302*2 = 6.697 + 0.6604 = 7.3574
    // Dim 1: 0.6697*5 + 0.3302*20 = 3.3485 + 6.604 = 9.9525
    
    let output_data = output.into_data().convert::<f32>().to_vec::<f32>().unwrap();
    println!("Output Data: {:?}", output_data);
    
    // Check values roughly
    let tolerance = 2e-3;
    assert!((output_data[0] - 7.357).abs() < tolerance, "Expected ~7.357, got {}", output_data[0]);
    assert!((output_data[1] - 9.952).abs() < tolerance, "Expected ~9.952, got {}", output_data[1]); 
}

