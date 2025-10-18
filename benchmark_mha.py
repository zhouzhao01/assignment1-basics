"""
Detailed profiling of custom multihead_self_attention implementation.
This script benchmarks each stage of the MHA forward pass to identify bottlenecks.
"""

import torch
import time
from basic_blocks.basic_blocks import multihead_self_attention, multihead_self_attention_fast

def profile_custom_mha(batch_size=8, seq_len=256, d_model=512, num_heads=16, num_iters=100, use_rope=True):
    """Profile custom MHA implementation with detailed timing"""

    device = 'cuda'

    # Create model
    mha = multihead_self_attention(
        d_model=d_model,
        num_heads=num_heads,
        max_seq_len=seq_len,
        theta=10000.0 if use_rope else None,
        device=device
    ).to(device)

    # Create input
    x = torch.randn(batch_size, seq_len, d_model, device=device)

    # Warmup
    for _ in range(10):
        _ = mha(x, flag_mask=True)

    # Detailed profiling
    timings = {
        'qkv_projection': 0.0,
        'rope_application': 0.0,
        'reshape_transpose': 0.0,
        'mask_creation': 0.0,
        'sdpa': 0.0,
        'output_projection': 0.0,
        'total': 0.0
    }

    d_k = d_model // num_heads

    for _ in range(num_iters):
        torch.cuda.synchronize()
        iter_start = time.time()

        # Stage 1: QKV Projection
        torch.cuda.synchronize()
        t0 = time.time()
        W_Q_in = torch.einsum("k d, ... s d -> ... s k", mha.W_Q, x)
        W_K_in = torch.einsum("k d, ... s d -> ... s k", mha.W_K, x)
        W_V_in = torch.einsum("v d, ... s d -> ... s v", mha.W_V, x)
        torch.cuda.synchronize()
        timings['qkv_projection'] += time.time() - t0

        # Stage 2: RoPE Application (if enabled)
        torch.cuda.synchronize()
        t0 = time.time()
        if mha.theta is not None:
            for head in range(num_heads):
                W_Q_in[..., head * d_k: (head + 1) * d_k] = mha.RoPE(
                    W_Q_in[..., head * d_k: (head + 1) * d_k],
                    torch.arange(0, seq_len, 1)
                )
                W_K_in[..., head * d_k: (head + 1) * d_k] = mha.RoPE(
                    W_K_in[..., head * d_k: (head + 1) * d_k],
                    torch.arange(0, seq_len, 1)
                )
        torch.cuda.synchronize()
        timings['rope_application'] += time.time() - t0

        # Stage 3: Reshape and Transpose
        torch.cuda.synchronize()
        t0 = time.time()
        Q = W_Q_in.view(batch_size, seq_len, num_heads, d_k).transpose(1, 2)
        K = W_K_in.view(batch_size, seq_len, num_heads, d_k).transpose(1, 2)
        V = W_V_in.view(batch_size, seq_len, num_heads, d_k).transpose(1, 2)
        torch.cuda.synchronize()
        timings['reshape_transpose'] += time.time() - t0

        # Stage 4: Mask Creation
        torch.cuda.synchronize()
        t0 = time.time()
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0).expand(batch_size, num_heads, -1, -1)
        torch.cuda.synchronize()
        timings['mask_creation'] += time.time() - t0

        # Stage 5: Scaled Dot Product Attention
        torch.cuda.synchronize()
        t0 = time.time()
        attentions = mha.SDPA(Q, K, V, mask)
        torch.cuda.synchronize()
        timings['sdpa'] += time.time() - t0

        # Stage 6: Output Projection
        torch.cuda.synchronize()
        t0 = time.time()
        attentions = attentions.transpose(1, 2).contiguous()
        attentions = attentions.view(batch_size, seq_len, num_heads * d_k)
        output = torch.einsum("d v, ... s v -> ... s d", mha.W_O, attentions)
        torch.cuda.synchronize()
        timings['output_projection'] += time.time() - t0

        torch.cuda.synchronize()
        timings['total'] += time.time() - iter_start

    # Average timings
    for key in timings:
        timings[key] /= num_iters

    return timings


def profile_fast_mha(batch_size=8, seq_len=256, d_model=512, num_heads=16, num_iters=100):
    """Profile PyTorch native MHA for comparison"""

    device = 'cuda'

    mha_fast = multihead_self_attention_fast(
        d_model=d_model,
        num_heads=num_heads,
        max_seq_len=seq_len,
        device=device
    ).to(device)

    x = torch.randn(batch_size, seq_len, d_model, device=device)

    # Warmup
    for _ in range(10):
        _ = mha_fast(x, flag_RoPE=False, flag_mask=True)

    # Benchmark
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iters):
        _ = mha_fast(x, flag_RoPE=False, flag_mask=True)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    return elapsed / num_iters


if __name__ == "__main__":
    # Benchmark with RoPE
    print("="*70)
    print("Profiling Custom MHA Implementation (WITH RoPE)")
    print("="*70)

    custom_timings_rope = profile_custom_mha(use_rope=True)
    fast_time = profile_fast_mha()

    print(f"\n{'Stage':<30} {'Time (ms)':<15} {'Percentage':<15}")
    print("-"*70)

    total_time_rope = custom_timings_rope['total']
    for stage, time_val in custom_timings_rope.items():
        if stage == 'total':
            continue
        time_ms = time_val * 1000
        percentage = (time_val / total_time_rope * 100) if total_time_rope > 0 else 0
        print(f"{stage:<30} {time_ms:<15.4f} {percentage:<15.2f}%")

    print("-"*70)
    print(f"{'Total':<30} {total_time_rope * 1000:<15.4f} {'100.00%':<15}")
    print("="*70)

    # Benchmark without RoPE
    print("\n")
    print("="*70)
    print("Profiling Custom MHA Implementation (WITHOUT RoPE)")
    print("="*70)

    custom_timings_no_rope = profile_custom_mha(use_rope=False)

    print(f"\n{'Stage':<30} {'Time (ms)':<15} {'Percentage':<15}")
    print("-"*70)

    total_time_no_rope = custom_timings_no_rope['total']
    for stage, time_val in custom_timings_no_rope.items():
        if stage == 'total':
            continue
        time_ms = time_val * 1000
        percentage = (time_val / total_time_no_rope * 100) if total_time_no_rope > 0 else 0
        print(f"{stage:<30} {time_ms:<15.4f} {percentage:<15.2f}%")

    print("-"*70)
    print(f"{'Total':<30} {total_time_no_rope * 1000:<15.4f} {'100.00%':<15}")
    print("="*70)

    # Comparison
    print("\n")
    print("="*70)
    print("Comparison Summary")
    print("="*70)
    print(f"Custom MHA with RoPE:    {total_time_rope * 1000:.4f} ms")
    print(f"Custom MHA without RoPE: {total_time_no_rope * 1000:.4f} ms")
    print(f"PyTorch Native MHA:      {fast_time * 1000:.4f} ms")
    print(f"\nRoPE overhead: {(total_time_rope - total_time_no_rope) * 1000:.4f} ms ({(total_time_rope - total_time_no_rope) / total_time_rope * 100:.2f}%)")
    print(f"Speedup (with RoPE):    {total_time_rope / fast_time:.2f}x slower than PyTorch native")
    print(f"Speedup (without RoPE): {total_time_no_rope / fast_time:.2f}x slower than PyTorch native")
    print("="*70)
