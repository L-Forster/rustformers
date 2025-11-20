import torch
import torch.nn as nn
import time

def run_parity():
    # PyTorch defaults to Float32 (fp32)
    # To use fp16 or bf16, you would typically use .half() or .bfloat16() 
    # or run inside `with torch.autocast(device_type=...):`
    
    d_model = 512
    n_heads = 8
    batch_size = 4
    seq_len = 128
    
    print(f"Initializing PyTorch MultiheadAttention(embed_dim={d_model}, num_heads={n_heads}, batch_first=True)")
    # Note: PyTorch defaults batch_first=False, so we set it to True to match our Rust implementation
    mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
    mha.eval() # Disable dropout for deterministic timing/parity check
    
    # Create random input
    # Shape: [batch_size, seq_len, d_model]
    x = torch.randn(batch_size, seq_len, d_model)
    
    print(f"Input shape: {x.shape}")
    
    start_time = time.time()
    # PyTorch forward returns (attn_output, attn_output_weights)
    attn_output, attn_weights = mha(x, x, x, need_weights=False)
    end_time = time.time()
    
    print(f"Output shape: {attn_output.shape}")
    print(f"Execution time: {(end_time - start_time) * 1000:.4f} ms")

if __name__ == "__main__":
    run_parity()

