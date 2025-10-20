import torch
import torch.nn as nn

class RoPE(nn.Module):
    def __init__(self, theta:float, d_k:int, max_seq_len:int, device=None):
        super().__init__()

        self.theta = theta
        self.d_k = d_k
        self.device = device

        # TODO: OPTIMIZATION STEP 1 - Replace rotation matrices with sin/cos caches
        # Current approach: O(max_seq_len × d_k²) memory for full matrices
        self.rotation_matrix = torch.zeros(max_seq_len,d_k,d_k, device=device)
        for seq_positon in range(max_seq_len):
            self.rotation_matrix[seq_positon,...] = self.cal_rotation_per_position(seq_positon)

        # TODO: OPTIMIZATION STEP 2 - Pre-compute only sin and cos values
        # Suggested implementation:
        # 1. Create frequency vector: freqs = 1.0 / (theta ** (torch.arange(0, d_k, 2) / d_k))
        # 2. Create position vector: positions = torch.arange(max_seq_len)
        # 3. Compute angles: angles = positions.unsqueeze(1) * freqs.unsqueeze(0)
        # 4. Pre-compute cos and sin: cos_cache = angles.cos(), sin_cache = angles.sin()
        # 5. Use register_buffer to store them:
        #    self.register_buffer('cos_cache', cos_cache, persistent=False)
        #    self.register_buffer('sin_cache', sin_cache, persistent=False)
        # This reduces memory to O(max_seq_len × d_k/2)

    def cal_rotation_per_position(self, token_position:int):
        # TODO: OPTIMIZATION STEP 5 - This method can be removed in optimized version
        # In the optimized implementation, you won't need to build full rotation matrices
        # The sin/cos caches in __init__ will replace this functionality
        rotation_matrix = torch.zeros(self.d_k, self.d_k, device=self.device)

        for k in torch.arange(1, self.d_k/2+1, 1):
            theta_i_d =  token_position / (self.theta ** ((2 * (k - 1)) / self.d_k))
            rotation_subblock_element = torch.tensor(
                [
                    [torch.cos(theta_i_d), -torch.sin(theta_i_d)],
                    [torch.sin(theta_i_d),  torch.cos(theta_i_d)]
                ],
                dtype=torch.float32,
                device= self.device
            )
            # k: 1, 2, 3, ...
            left = 2 * (k.to(torch.int) - 1)   # left: 0, 2, 4, ...
            right = 2 * k.to(torch.int)     # right: 1, 3, 5, ...
            rotation_matrix[left:right, left:right] = rotation_subblock_element

        return rotation_matrix

    # TODO: OPTIMIZATION STEP 6 - Add helper method for applying rotation (optional)
    # def _apply_rotation(self, x, cos, sin):
    #     """Apply rotation to x using precomputed cos and sin values."""
    #     # Reshape to separate feature pairs
    #     x_reshape = x.reshape(*x.shape[:-1], -1, 2)
    #     # Apply rotation
    #     x_rot = torch.empty_like(x_reshape)
    #     x_rot[..., 0] = x_reshape[..., 0] * cos - x_reshape[..., 1] * sin
    #     x_rot[..., 1] = x_reshape[..., 0] * sin + x_reshape[..., 1] * cos
    #     # Reshape back
    #     return x_rot.reshape(*x.shape)
    
    def forward(self, x:torch.Tensor, token_position:torch.Tensor) -> torch.Tensor:
        # TODO: OPTIMIZATION STEP 3 - Replace matrix multiplication with element-wise ops
        # Current approach: Loop through positions and apply matrix multiplication
        # x = x.to(self.device)
        # breakpoint()

        # Debug assertions
        max_pos = token_position.max().item()
        assert max_pos < self.rotation_matrix.shape[0], \
            f"RoPE Error: position {max_pos} exceeds rotation_matrix size {self.rotation_matrix.shape[0]}. " \
            f"x.shape={x.shape}, token_position range=[{token_position.min().item()}, {max_pos}]"

        # Check for NaN/Inf in input
        if torch.isnan(x).any():
            raise ValueError(f"RoPE: NaN detected in input x! Shape: {x.shape}")
        if torch.isinf(x).any():
            raise ValueError(f"RoPE: Inf detected in input x! Shape: {x.shape}")

        for position in token_position:
            """
            Be careful. It should be x * R.T.

            x_ROPE = R @ x -> x
            x_ROPE^T = (R @ x)^T = x^T @ R^T
            """
            position = position.item()
            x[...,position,:] =   x[...,position,:]  @ self.rotation_matrix[position,...].T

        # TODO: OPTIMIZATION STEP 4 - Optimized forward pass
        # Suggested implementation:
        # 1. Reshape x to separate pairs: x_reshape = x.reshape(*x.shape[:-1], -1, 2)
        # 2. Get cos/sin for needed positions:
        #    cos = self.cos_cache[token_position]  # Shape: [seq_len, d_k//2]
        #    sin = self.sin_cache[token_position]  # Shape: [seq_len, d_k//2]
        # 3. Expand cos/sin to match x's batch dimensions if needed
        # 4. Apply rotation to pairs:
        #    x_rot = torch.empty_like(x_reshape)
        #    x_rot[..., 0] = x_reshape[..., 0] * cos - x_reshape[..., 1] * sin
        #    x_rot[..., 1] = x_reshape[..., 0] * sin + x_reshape[..., 1] * cos
        # 5. Reshape back: return x_rot.reshape(*x.shape)
        # This avoids loops and uses efficient element-wise operations

        return x

class RoPE_fast(nn.Module):
    def __init__(self, theta:float, d_k:int, max_seq_len:int, device=None):
        super().__init__()

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        self.cos_cached = None
        self.sin_cached = None

    def _build_cache(self, x:torch.Tensor):
        if self.cos_cached is not None and self.sin_cached is not None:
            return
        
        position = torch.arange(0, self.max_seq_len, device=self.device)

        k = torch.arange(0, self.d_k, 2, device=self.device) / self.d_k
        theta = self.theta ** -( k )
        

        theta_matrix = torch.einsum("l, k -> lk", position, theta)

        self.cos_cached = torch.cos(theta_matrix).repeat_interleave(2,dim=-1)
        self.sin_cached = torch.sin(theta_matrix).repeat_interleave(2,dim=-1)
        

    def forward(self, x:torch.Tensor):
        self._build_cache(x)

        seq_len = x.shape[-2] # deal with case which len(input_feature) is shorter than max_len 

        x_sin = self.reorder_sequence(x)
        return x * self.cos_cached[:seq_len,:] + x_sin * self.sin_cached[:seq_len,:]

    def reorder_sequence(self, x: torch.Tensor):
        D = x.shape[-1]

        # [-1, 1, -1 , 1, ...]
        sign = torch.ones(D, device=x.device, dtype=x.dtype)
        sign[0::2] = -1

        # [..., D] -> [..., D//2, 2]
        x_reshape = x.reshape(*x.shape[:-1], D//2 , 2)
        # 翻转最后一维
        x_flipped = x_reshape.flip(-1)
        # [..., D//2, 2] -> [..., D]
        x = x_flipped.reshape(*x.shape[:-1], D)
        return x * sign

def benchmark_rope_implementations(
    batch_sizes=[8, 16, 32],
    seq_lens=[2048, 4096],
    d_ks=[64, 128],
    theta=10000.0,
    num_warmup=10,
    num_iterations=100,
    device='cuda'
):
    """
    比较 RoPE 和 RoPE_fast 的性能

    Args:
        batch_sizes: 批次大小列表
        seq_lens: 序列长度列表
        d_ks: 头维度列表
        theta: RoPE 的 theta 参数
        num_warmup: 预热次数
        num_iterations: 测试迭代次数
        device: 设备

    Returns:
        results: 包含每个配置的性能数据的列表
    """
    import time

    results = []

    print(f"{'Config':<30} {'RoPE (ms)':<15} {'RoPE_fast (ms)':<15} {'Speedup':<10} {'Memory RoPE (MB)':<20} {'Memory Fast (MB)':<20}")
    print("=" * 120)

    for batch_size in batch_sizes:
        for seq_len in seq_lens:
            for d_k in d_ks:
                config_name = f"B{batch_size}_S{seq_len}_D{d_k}"

                # 初始化两个模型
                rope_slow = RoPE(theta=theta, d_k=d_k, max_seq_len=seq_len, device=device)
                rope_fast = RoPE_fast(theta=theta, d_k=d_k, max_seq_len=seq_len, device=device)

                # 准备测试数据
                x_slow = torch.randn(batch_size, seq_len, d_k, device=device)
                x_fast = torch.randn(batch_size, seq_len, d_k, device=device)
                token_positions = torch.arange(seq_len, device=device)

                # ===== 测试 RoPE (慢版本) =====
                # 预热
                for _ in range(num_warmup):
                    _ = rope_slow(x_slow.clone(), token_positions)

                torch.cuda.synchronize()

                # 计时
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

                start_event.record()
                for _ in range(num_iterations):
                    _ = rope_slow(x_slow.clone(), token_positions)
                end_event.record()

                torch.cuda.synchronize()
                time_slow = start_event.elapsed_time(end_event) / num_iterations  # ms

                # 内存使用（粗略估计）
                mem_slow = rope_slow.rotation_matrix.numel() * rope_slow.rotation_matrix.element_size() / 1024 / 1024  # MB

                # ===== 测试 RoPE_fast =====
                # 预热
                for _ in range(num_warmup):
                    _ = rope_fast(x_fast.clone())

                torch.cuda.synchronize()

                # 计时
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

                start_event.record()
                for _ in range(num_iterations):
                    _ = rope_fast(x_fast.clone())
                end_event.record()

                torch.cuda.synchronize()
                time_fast = start_event.elapsed_time(end_event) / num_iterations  # ms

                # 内存使用
                mem_fast = (rope_fast.cos_cached.numel() + rope_fast.sin_cached.numel()) * rope_fast.cos_cached.element_size() / 1024 / 1024  # MB

                # 加速比
                speedup = time_slow / time_fast

                # 保存结果
                results.append({
                    'config': config_name,
                    'batch_size': batch_size,
                    'seq_len': seq_len,
                    'd_k': d_k,
                    'time_slow_ms': time_slow,
                    'time_fast_ms': time_fast,
                    'speedup': speedup,
                    'mem_slow_mb': mem_slow,
                    'mem_fast_mb': mem_fast
                })

                print(f"{config_name:<30} {time_slow:<15.4f} {time_fast:<15.4f} {speedup:<10.2f}x {mem_slow:<20.2f} {mem_fast:<20.2f}")

                # 清理内存
                del rope_slow, rope_fast, x_slow, x_fast
                torch.cuda.empty_cache()

    return results

if __name__ == "__main__":
    # 运行性能测试
    if torch.cuda.is_available():
        print("Running RoPE performance benchmark...\n")
        results = benchmark_rope_implementations(
            batch_sizes=[8, 16, 32],
            seq_lens=[2048, 4096, 8192],
            d_ks=[64, 128],
            num_warmup=10,
            num_iterations=100,
            device='cuda'
        )

        print("\n" + "=" * 120)
        print("Benchmark complete!")
        print(f"Average speedup: {sum(r['speedup'] for r in results) / len(results):.2f}x")
    else:
        print("CUDA not available. Skipping benchmark.")
