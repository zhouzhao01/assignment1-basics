import torch
import time

def benchmark_matmul(size=8192, dtype=torch.float32, num_iterations=100, warmup=10):
    """
    Benchmark matrix multiplication to estimate GPU FLOPS.
    
    Args:
        size: Matrix dimension (will compute size x size @ size x size)
        dtype: Data type (torch.float32, torch.float16, torch.bfloat16)
        num_iterations: Number of iterations to average
        warmup: Number of warmup iterations
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if device.type == 'cpu':
        print("Warning: CUDA not available, running on CPU")
        return None
    
    # Print GPU info
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"Matrix size: {size}x{size}")
    print(f"Data type: {dtype}")
    print("-" * 60)
    
    # Create random matrices
    A = torch.randn(size, size, dtype=dtype, device=device)
    B = torch.randn(size, size, dtype=dtype, device=device)
    
    # Warmup
    for _ in range(warmup):
        C = torch.matmul(A, B)
    torch.cuda.synchronize()
    
    # Benchmark
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(num_iterations):
        C = torch.matmul(A, B)
    end_event.record()
    
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    
    # Calculate FLOPS
    # Matrix multiplication of (M x K) @ (K x N) = M * N * (2K - 1) ≈ 2MNK operations
    flops_per_matmul = 2 * size ** 3
    total_flops = flops_per_matmul * num_iterations
    
    elapsed_time_s = elapsed_time_ms / 1000.0
    tflops = (total_flops / elapsed_time_s) / 1e12
    
    avg_time_ms = elapsed_time_ms / num_iterations
    
    print(f"Average time per matmul: {avg_time_ms:.3f} ms")
    print(f"Achieved performance: {tflops:.2f} TFLOPS")
    
    return tflops

def get_theoretical_flops():
    """
    Get theoretical FLOPS for common GPU models.
    Note: These are approximate FP32 values.
    """
    gpu_specs = {
        # Consumer GPUs
        "RTX 4090": 82.6,
        "RTX 4080": 48.7,
        "RTX 4070 Ti": 40.1,
        "RTX 3090": 35.6,
        "RTX 3080": 29.8,
        "RTX 3070": 20.3,
        # Professional GPUs
        "A100": 19.5,  # FP32, but 312 TFLOPS for FP16 with Tensor Cores
        "H100": 51.2,  # FP32, but 1979 TFLOPS for FP8 with Tensor Cores
        "V100": 14.0,
        "A6000": 38.7,
    }
    
    gpu_name = torch.cuda.get_device_name(0)
    
    print("\nTheoretical Peak FLOPS (FP32):")
    print("-" * 60)
    
    for name, tflops in gpu_specs.items():
        if name in gpu_name:
            print(f"Your GPU ({name}): ~{tflops} TFLOPS (theoretical)")
            return tflops
    
    print("GPU not in database. Look up your GPU specifications online.")
    print("Calculate as: Cores × Clock Speed (GHz) × 2 / 1000 = TFLOPS")
    return None

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA is not available. Please ensure you have a CUDA-capable GPU and PyTorch with CUDA support.")
        exit(1)
    
    print("=" * 60)
    print("GPU FLOPS Benchmark")
    print("=" * 60)
    print()
    
    # Get theoretical specs
    theoretical = get_theoretical_flops()
    print()
    
    # Run benchmarks with different precisions
    print("Running FP32 benchmark...")
    fp32_tflops = benchmark_matmul(size=8192, dtype=torch.float32, num_iterations=100)
    print()
    
    # FP16 benchmark (if supported)
    if torch.cuda.get_device_capability()[0] >= 7:  # Volta and newer
        print("Running FP16 benchmark...")
        fp16_tflops = benchmark_matmul(size=8192, dtype=torch.float16, num_iterations=100)
        print()
    
    # Test with larger matrices for more compute-bound scenario
    print("Running larger matrix test (FP32)...")
    large_fp32_tflops = benchmark_matmul(size=16384, dtype=torch.float32, num_iterations=20)
    print()
    
    print("=" * 60)
    print("Summary:")
    print("-" * 60)
    if theoretical:
        efficiency = (fp32_tflops / theoretical) * 100
        print(f"Theoretical Peak (FP32): {theoretical:.2f} TFLOPS")
        print(f"Achieved (FP32): {fp32_tflops:.2f} TFLOPS ({efficiency:.1f}% efficiency)")
    else:
        print(f"Achieved (FP32): {fp32_tflops:.2f} TFLOPS")
    print("=" * 60)
