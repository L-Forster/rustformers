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

    let output = mha.compute_multi_head_attention(q, k, v, None);
    
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
    let output = mha.compute_multi_head_attention(x.clone(), x.clone(), x.clone(), None);
    
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
    let output = super::scaled_dot_product_attention(q, k, v, None, &dropout);
    
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

#[test]
fn test_scaled_dot_product_attention_masking() {
    let device = Default::default();
    
    // Setup inputs: q0, q1 identical. k0, k1 identical. v0=[1,2], v1=[3,4].
    let q_data: [f32; 4] = [1.0, 0.0, 1.0, 0.0]; 
    let k_data: [f32; 4] = [1.0, 0.0, 1.0, 0.0];
    let v_data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    
    let q = Tensor::<TestBackend, 1>::from_floats(q_data, &device).reshape([1, 1, 2, 2]);
    let k = Tensor::<TestBackend, 1>::from_floats(k_data, &device).reshape([1, 1, 2, 2]);
    let v = Tensor::<TestBackend, 1>::from_floats(v_data, &device).reshape([1, 1, 2, 2]);
    
    // Mask out index 1 for query 0.
    // mask shape [1, 1, 2, 2]
    // q0 attends to k0, k1. Mask k1 (index 1).
    // q1 attends to k0, k1. No mask.
    let mask_data: [f32; 4] = [0.0, -1.0e9, 0.0, 0.0];
    let mask = Tensor::<TestBackend, 1>::from_floats(mask_data, &device).reshape([1, 1, 2, 2]);
    
    let dropout = burn::nn::DropoutConfig::new(0.0).init();
    
    let output = super::scaled_dot_product_attention(q, k, v, Some(mask), &dropout);
    
    let output_data = output.into_data().convert::<f32>().to_vec::<f32>().unwrap();
    println!("Masked Output: {:?}", output_data);
    
    // Row 0: q0 attends only to k0. Output should be v0 = [1.0, 2.0].
    assert!((output_data[0] - 1.0).abs() < 1e-3, "Expected 1.0, got {}", output_data[0]);
    assert!((output_data[1] - 2.0).abs() < 1e-3, "Expected 2.0, got {}", output_data[1]);
    
    // Row 1: q1 attends to k0 and k1 equally. Output should be average of v0 and v1 = [2.0, 3.0].
    assert!((output_data[2] - 2.0).abs() < 1e-3, "Expected 2.0, got {}", output_data[2]);
    assert!((output_data[3] - 3.0).abs() < 1e-3, "Expected 3.0, got {}", output_data[3]);
}
