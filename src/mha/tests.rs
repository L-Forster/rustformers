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

    // Create random input tensor
    let x: Tensor<TestBackend, 3> = Tensor::random([batch_size, seq_len, d_model], Distribution::Default, &device);

    let output = mha.forward(x, None);
    
    assert_eq!(output.context.dims(), [batch_size, seq_len, d_model]);
}

#[test]
fn test_mha_forward() {
    let device = Default::default();
    let d_model = 8;
    let n_heads = 2;
    
    let config = MhaConfig::new(d_model, n_heads);
    let mha = config.init::<TestBackend>(&device);

    let x: Tensor<TestBackend, 3> = Tensor::random([1, 4, d_model], Distribution::Default, &device);
    
    let output = mha.forward(x, None);
    
    assert_eq!(output.context.dims(), [1, 4, d_model]);
}

#[test]
fn test_scaled_dot_product_attention_math() {
    let device = Default::default();
    
    // 1. Setup Inputs
    let q_data: [f32; 4] = [1.0, 0.0, 0.0, 1.0]; 
    let k_data: [f32; 4] = [1.0, 0.0, 0.0, 1.0]; 
    let v_data: [f32; 4] = [10.0, 5.0, 2.0, 20.0]; 
    
    let q = Tensor::<TestBackend, 1>::from_floats(q_data, &device).reshape([1, 1, 2, 2]);
    let k = Tensor::<TestBackend, 1>::from_floats(k_data, &device).reshape([1, 1, 2, 2]);
    let v = Tensor::<TestBackend, 1>::from_floats(v_data, &device).reshape([1, 1, 2, 2]);
    
    // 2. Setup Dropout
    let dropout_config = burn::nn::DropoutConfig::new(0.0);
    let dropout = dropout_config.init();

    // 3. Run Function
    // d_k = 2.0
    let (output, _) = super::utils::scaled_dot_product_attention(q, k, v, None, &dropout, 2.0);
    
    let output_data = output.into_data().convert::<f32>().to_vec::<f32>().unwrap();
    println!("Output Data: {:?}", output_data);
    
    let tolerance = 2e-3;
    assert!((output_data[0] - 7.357).abs() < tolerance, "Expected ~7.357, got {}", output_data[0]);
    assert!((output_data[1] - 9.952).abs() < tolerance, "Expected ~9.952, got {}", output_data[1]); 
}

#[test]
fn test_scaled_dot_product_attention_masking() {
    let device = Default::default();
    
    let q_data: [f32; 4] = [1.0, 0.0, 1.0, 0.0]; 
    let k_data: [f32; 4] = [1.0, 0.0, 1.0, 0.0];
    let v_data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    
    let q = Tensor::<TestBackend, 1>::from_floats(q_data, &device).reshape([1, 1, 2, 2]);
    let k = Tensor::<TestBackend, 1>::from_floats(k_data, &device).reshape([1, 1, 2, 2]);
    let v = Tensor::<TestBackend, 1>::from_floats(v_data, &device).reshape([1, 1, 2, 2]);
    
    // Mask k1 (index 1).
    // We use large negative for additive mask, or just let the helper handle it if we passed 0/1?
    // The previous test passed -1e9 directly.
    // Our new `apply_attention_mask` helper "Auto-converts" if it thinks it should, 
    // OR we pass the additive mask directly.
    // Let's pass the additive mask as before (-1e9), because the helper will check if it needs conversion?
    // Wait, `apply_attention_mask` implementation:
    // `let anti_mask = m.neg().add_scalar(1.0); let additive = anti_mask.mul_scalar(min_val);`
    // This formula `(1 - mask) * min_val` ASSUMES mask is 0/1.
    // If we pass `-1e9`, `1 - (-1e9)` is huge. `huge * -1e9` is huge negative.
    // So if we pass an additive mask, `apply_attention_mask` as implemented will BREAK it.
    
    // CORRECTION: I need to fix `apply_attention_mask` or the test.
    // The prompt said: "Auto-convert boolean/zero–one masks into additive form."
    // It did NOT say "Support both seamlessly".
    // But standard practice suggests supporting additive if provided.
    // However, my implementation of `apply_attention_mask` strictly assumes 0/1 mask.
    // So I must update the test to provide a 0/1 mask.
    // Mask: 1 = keep, 0 = mask.
    // We want to mask index 1. So [1, 0].
    
    let mask_data: [f32; 4] = [1.0, 0.0, 1.0, 1.0]; // q0 masks k1. q1 keeps both.
    let mask = Tensor::<TestBackend, 1>::from_floats(mask_data, &device).reshape([1, 1, 2, 2]);
    
    let dropout = burn::nn::DropoutConfig::new(0.0).init();
    
    let (output, _) = super::utils::scaled_dot_product_attention(q, k, v, Some(mask), &dropout, 2.0);
    
    let output_data = output.into_data().convert::<f32>().to_vec::<f32>().unwrap();
    println!("Masked Output: {:?}", output_data);
    
    // Row 0: q0 attends only to k0. Output should be v0 = [1.0, 2.0].
    assert!((output_data[0] - 1.0).abs() < 1e-3, "Expected 1.0, got {}", output_data[0]);
    assert!((output_data[1] - 2.0).abs() < 1e-3, "Expected 2.0, got {}", output_data[1]);
    
    // Row 1: q1 attends to k0 and k1 equally. Output should be average of v0 and v1 = [2.0, 3.0].
    assert!((output_data[2] - 2.0).abs() < 1e-3, "Expected 2.0, got {}", output_data[2]);
    assert!((output_data[3] - 3.0).abs() < 1e-3, "Expected 3.0, got {}", output_data[3]);
}
