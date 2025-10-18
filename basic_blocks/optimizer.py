# Custom AdamW Optimizer for CS336.
# Zhao Zhou
# 2025.9.26

import torch
import torch.nn as nn

from  collections.abc import Callable, Iterable
from typing import Optional


import math

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, weight_decay=0.01, betas = (0.9, 0.999), eps=1e-8                ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if betas[0] < 0:
            raise ValueError(f"Invalid betas[0]: {betas[0]}")
        if betas[1] < 0:
            raise ValueError(f"Invalid betas[1]: {betas}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")   
        
        self.eps = eps

        defaults = {
            "alpha": lr,
            "beta_1": betas[0],
            "beta_2": betas[1],
            "weight_decay": weight_decay,
        }

        super().__init__(params, defaults)

    def step(self, closure:Optional[Callable]=None):
        #  Step 1: Calculate loss using closure()
        loss = None if closure is None else closure()

        for group in self.param_groups: # for every param group
            # Cache group-level hyperparameters (avoid dict lookup in inner loop)
            beta_1 = group['beta_1']
            beta_2 = group['beta_2']
            alpha = group["alpha"]
            weight_decay = group["weight_decay"]

            for p in group["params"]: # for each param in one param group

                # Step 1: get specific grad for specific param
                if p.grad is None:
                    continue
                grad = p.grad

                # Step 2: Initialize State
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                # Cache state variables
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                step = state['step'] + 1

                # Step 3: Calculate 1st, 2nd Momentum (in-place operations)
                exp_avg.mul_(beta_1).add_(grad, alpha=1 - beta_1)
                exp_avg_sq.mul_(beta_2).addcmul_(grad, grad, value=1 - beta_2)

                # Step 4: Compute bias-corrected learning rate (pure tensor ops)
                bias_correction1 = 1 - beta_1 ** step
                bias_correction2 = 1 - beta_2 ** step
                step_size = alpha * (bias_correction2 ** 0.5) / bias_correction1

                # Step 5: Update parameter (in-place operations on .data)
                # First apply weight decay
                p.data.mul_(1 - alpha * weight_decay)
                # Then apply Adam update
                p.data.addcdiv_(exp_avg, exp_avg_sq.sqrt().add_(self.eps), value=-step_size)

                state["step"] = step
        return loss

def get_lr_cosine_schedule(t: int, max_learning_rate: float, min_learning_rate: float, warmup_iters: int, cosine_cycle_iters: int) -> float:
    """
    Calculate learning rate at iteration t using cosine schedule with warmup.

    Args:
        t (int): Current iteration number
        max_learning_rate (float): Maximum learning rate (alpha_max)
        min_learning_rate (float): Minimum learning rate (alpha_min)
        warmup_iters (int): Number of warmup iterations (T_w)
        cosine_cycle_iters (int): Total iterations for cosine cycle (T_c)

    Returns:
        float: Learning rate at iteration t
    """
    if t < 0:
        raise ValueError(f"Invalid learning rate step: {t}")
    elif t < warmup_iters:
        # Linear warmup: 0 → max_lr
        lr = t / warmup_iters * max_learning_rate
    elif t < cosine_cycle_iters:
        # Cosine decay: max_lr → min_lr
        lr = min_learning_rate + 0.5 * (1 + math.cos((t - warmup_iters) / (cosine_cycle_iters - warmup_iters) * math.pi)) * (max_learning_rate - min_learning_rate)
    else:
        # Stay at minimum
        lr = min_learning_rate
    return lr


class lr_cosine_schedule(nn.Module):
    """Learning rate scheduler that applies cosine schedule with warmup to an optimizer."""

    def __init__(self, optimizer: torch.optim.Optimizer, max_learning_rate: float, min_learning_rate: float, warmup_iters: int, cosine_cycle_iters: int):
        super().__init__()
        self.optimizer = optimizer
        self.max_learning_rate = max_learning_rate
        self.min_learning_rate = min_learning_rate
        self.warmup_iters = warmup_iters
        self.cosine_cycle_iters = cosine_cycle_iters
        self.iteration = 0

    def get_lr(self, t: int) -> float:
        """Get learning rate at iteration t."""
        return get_lr_cosine_schedule(t, self.max_learning_rate, self.min_learning_rate, self.warmup_iters, self.cosine_cycle_iters)

    def step(self):
        """Update optimizer's learning rate and increment iteration counter."""
        # Inline the LR calculation to avoid function call overhead
        t = self.iteration
        if t < self.warmup_iters:
            current_lr = t / self.warmup_iters * self.max_learning_rate
        elif t < self.cosine_cycle_iters:
            current_lr = self.min_learning_rate + 0.5 * (1 + math.cos((t - self.warmup_iters) / (self.cosine_cycle_iters - self.warmup_iters) * math.pi)) * (self.max_learning_rate - self.min_learning_rate)
        else:
            current_lr = self.min_learning_rate

        for group in self.optimizer.param_groups:
            group["alpha"] = current_lr
        self.iteration += 1

class grad_clip(nn.Module):
    def __init__(self, max_l2_norm:float, eps:float=1e-6):
        """
        Args:
            parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
            max_l2_norm (float): a positive value containing the maximum l2-norm.
        """
        super().__init__()
        self.max_l2_norm = max_l2_norm
        self.eps = eps

    def forward(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.
        The gradients of the parameters (parameter.grad) should be modified in-place.
        """
        # Step 1: Compute total norm WITHOUT extracting gradients
        total_norm_squared = 0.0
        for each_param in parameters:
            if each_param.grad is not None:
                total_norm_squared += torch.linalg.norm(each_param.grad) ** 2
        
        # Step 2: Get the total norm
        total_norm = torch.sqrt(total_norm_squared)

        # Step 3: If clipping is needed, apply scaling factor to each gradient IN-PLACE
        if total_norm > self.max_l2_norm:
            scaling_factor = self.max_l2_norm / (total_norm + self.eps)
            for each_param in parameters:
                if each_param.grad is not None:
                    each_param.grad *= scaling_factor
    
    def cal_total_l2norm(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        # Step 1: Compute total norm WITHOUT extracting gradients
        total_norm_squared = 0.0
        for each_param in parameters:
            if each_param.grad is not None:
                total_norm_squared += torch.linalg.norm(each_param.grad) ** 2
        
        # Step 2: Get the total norm
        total_norm = torch.sqrt(total_norm_squared)

        return total_norm

def test_optimizer():
    """
    Test custom AdamW optimizer with cosine learning rate scheduler.
    Demonstrates warmup → cosine decay → minimum LR phases.
    """
    # Step 1: Create a simple toy model
    torch.manual_seed(42)
    model = torch.nn.Linear(3, 1, bias=False)  # Simple linear regression

    # Step 2: Create custom AdamW optimizer
    optimizer = AdamW(  # YOUR custom AdamW, not torch.optim.AdamW
        model.parameters(),
        lr=1e-4,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    # Step 3: Create custom learning rate scheduler
    lr_scheduler = lr_cosine_schedule(
        optimizer=optimizer,
        max_learning_rate=1e-4,
        min_learning_rate=1e-7,
        warmup_iters=100,
        cosine_cycle_iters=1000
    )

    # Step 4: Training loop
    print("Testing Custom AdamW + Cosine LR Scheduler")
    print("=" * 60)
    print(f"{'Iteration':>10} | {'Learning Rate':>15} | {'Loss':>10}")
    print("-" * 60)

    for iteration in range(1500):
        # Generate random training data
        x = torch.randn(10, 3)      # 10 samples, 3 features
        y_true = torch.randn(10, 1)  # 10 target values

        # Forward pass
        optimizer.zero_grad()
        y_pred = model(x)
        loss = ((y_pred - y_true) ** 2).mean()

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update learning rate (AFTER optimizer.step())
        lr_scheduler.step()

        # Print progress at key iterations to see LR schedule
        if iteration in [0, 50, 99, 100, 500, 999, 1000, 1499]:
            current_lr = optimizer.param_groups[0]['alpha']  # Use 'alpha', not 'lr'
            print(f"{iteration:>10} | {current_lr:>15.2e} | {loss.item():>10.4f}")

    print("=" * 60)
    print("\nExpected behavior:")
    print("  - Iterations 0-99:   Warmup (0 → 1e-4)")
    print("  - Iterations 100-999: Cosine decay (1e-4 → 1e-7)")
    print("  - Iterations 1000+:   Minimum LR (1e-7)")

def benchmark_optimizer_performance():
    """
    Performance benchmarking for custom AdamW and lr_cosine_schedule.
    Compares against PyTorch's official implementations.
    """
    import time
    from typing import Tuple

    def create_test_model(device='cuda'):
        """Create a moderately sized model for realistic performance testing."""
        model = nn.Sequential(
            nn.Linear(512, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
        ).to(device)
        return model

    def benchmark_single(
        optimizer_class,
        scheduler_class,
        model,
        num_iterations=1000,
        device='cuda',
    ) -> Tuple[float, float]:
        """Benchmark optimizer + scheduler performance."""
        # Reset model
        for p in model.parameters():
            p.data.normal_()

        # Create optimizer
        if optimizer_class == torch.optim.AdamW:
            optimizer = optimizer_class(
                model.parameters(),
                lr=1e-3,
                weight_decay=0.01,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        else:  # CustomAdamW
            optimizer = optimizer_class(
                model.parameters(),
                lr=1e-3,
                weight_decay=0.01,
                betas=(0.9, 0.999),
                eps=1e-8
            )

        # Create scheduler
        if scheduler_class == torch.optim.lr_scheduler.CosineAnnealingLR:
            scheduler = scheduler_class(optimizer, T_max=num_iterations)
        else:  # CustomScheduler
            scheduler = scheduler_class(
                optimizer,
                max_learning_rate=1e-3,
                min_learning_rate=1e-6,
                warmup_iters=100,
                cosine_cycle_iters=num_iterations
            )

        # Warmup (avoid cold start effects)
        x = torch.randn(64, 512, device=device)
        for _ in range(10):
            optimizer.zero_grad()
            loss = model(x).sum()
            loss.backward()
            optimizer.step()
            scheduler.step()

        # Synchronize GPU before timing
        if device == 'cuda':
            torch.cuda.synchronize()

        # Benchmark
        optimizer_times = []
        scheduler_times = []

        for i in range(num_iterations):
            x = torch.randn(64, 512, device=device)
            optimizer.zero_grad()
            loss = model(x).sum()
            loss.backward()

            # Time optimizer step
            if device == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            optimizer.step()
            if device == 'cuda':
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            optimizer_times.append(t1 - t0)

            # Time scheduler step
            t0 = time.perf_counter()
            scheduler.step()
            if device == 'cuda':
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            scheduler_times.append(t1 - t0)

        avg_optimizer_time = sum(optimizer_times) / len(optimizer_times)
        avg_scheduler_time = sum(scheduler_times) / len(scheduler_times)
        return avg_optimizer_time, avg_scheduler_time

    # Main benchmark
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running benchmarks on: {device}")
    print("=" * 80)

    num_iterations = 1000

    # Benchmark PyTorch
    print("\n[1/2] Benchmarking PyTorch AdamW + CosineAnnealingLR...")
    model_pytorch = create_test_model(device)
    opt_time_pytorch, sched_time_pytorch = benchmark_single(
        torch.optim.AdamW,
        torch.optim.lr_scheduler.CosineAnnealingLR,
        model_pytorch,
        num_iterations=num_iterations,
        device=device,
    )
    total_time_pytorch = opt_time_pytorch + sched_time_pytorch

    # Benchmark custom
    print("[2/2] Benchmarking Custom AdamW + lr_cosine_schedule...")
    model_custom = create_test_model(device)
    opt_time_custom, sched_time_custom = benchmark_single(
        AdamW,
        lr_cosine_schedule,
        model_custom,
        num_iterations=num_iterations,
        device=device,
    )
    total_time_custom = opt_time_custom + sched_time_custom

    # Print results
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS (average per iteration)")
    print("=" * 80)
    print(f"\n{'Component':<30} {'PyTorch':>15} {'Custom':>15} {'Slowdown':>15}")
    print("-" * 80)
    print(f"{'Optimizer step (ms)':<30} {opt_time_pytorch*1000:>15.4f} {opt_time_custom*1000:>15.4f} {opt_time_custom/opt_time_pytorch:>15.2f}x")
    print(f"{'Scheduler step (ms)':<30} {sched_time_pytorch*1000:>15.4f} {sched_time_custom*1000:>15.4f} {sched_time_custom/sched_time_pytorch:>15.2f}x")
    print(f"{'Total (ms)':<30} {total_time_pytorch*1000:>15.4f} {total_time_custom*1000:>15.4f} {total_time_custom/total_time_pytorch:>15.2f}x")
    print("=" * 80)
    print(f"\nTotal time for {num_iterations} iterations:")
    print(f"  PyTorch: {total_time_pytorch * num_iterations:.3f}s")
    print(f"  Custom:  {total_time_custom * num_iterations:.3f}s")
    print(f"  Overhead: {(total_time_custom - total_time_pytorch) * num_iterations:.3f}s")


def benchmark_with_real_model():
    """
    Benchmark with actual transformer_lm model using config from config_4090.json
    """
    import sys
    import time
    from pathlib import Path

    # Add basic_blocks to path if needed
    current_dir = Path(__file__).parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))

    from scaffoldings import builder

    config_path = "/mnt/aat/zzhao.zhou/cs336_2025/assignment1-basics/configs/config_4090.json"

    print("=" * 80)
    print("REAL MODEL BENCHMARK: Custom AdamW vs PyTorch AdamW")
    print("=" * 80)
    print(f"\nLoading config from: {config_path}")

    # Build model using config
    config_builder = builder(config_path)

    def run_benchmark(use_custom_optimizer=True, num_iterations=100):
        """Run benchmark with either custom or PyTorch optimizer"""
        # Build fresh model
        model = config_builder.build_model()
        print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Build optimizer
        opt_config = config_builder.config['optimizer']
        if use_custom_optimizer:
            optimizer = AdamW(
                params=model.parameters(),
                lr=opt_config['lr'],
                betas=tuple(opt_config['betas']),
                weight_decay=opt_config['weight_decay'],
                eps=opt_config['eps']
            )
            opt_name = "Custom AdamW"
        else:
            optimizer = torch.optim.AdamW(
                params=model.parameters(),
                lr=opt_config['lr'],
                betas=tuple(opt_config['betas']),
                eps=opt_config['eps'],
                weight_decay=opt_config['weight_decay']
            )
            opt_name = "PyTorch AdamW"

        # Build scheduler
        lr_config = config_builder.config['lr_scheduler']
        total_batches = config_builder.calculate_total_batches()

        if use_custom_optimizer:
            scheduler = lr_cosine_schedule(
                optimizer=optimizer,
                max_learning_rate=lr_config['max_learning_rate'],
                min_learning_rate=lr_config['min_learning_rate'],
                warmup_iters=lr_config['warmup_iters'],
                cosine_cycle_iters=total_batches
            )
            sched_name = "Custom lr_cosine_schedule"
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                T_max=total_batches
            )
            sched_name = "PyTorch CosineAnnealingLR"

        print(f"\n[Testing {opt_name} + {sched_name}]")

        # Get config values
        batch_size = config_builder.config['training']['batch_size']
        context_length = config_builder.config['model']['context_length']
        vocab_size = config_builder.config['model']['vocab_size']
        device = config_builder.config['device']

        # Warmup
        for _ in range(5):
            input_ids = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
            optimizer.zero_grad()
            logits = model(input_ids)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, vocab_size),
                input_ids.reshape(-1)
            )
            loss.backward()
            optimizer.step()
            scheduler.step()

        # Benchmark
        if device == 'cuda':
            torch.cuda.synchronize()

        optimizer_times = []
        scheduler_times = []
        forward_times = []
        backward_times = []

        for _ in range(num_iterations):
            input_ids = torch.randint(0, vocab_size, (batch_size, context_length), device=device)

            # Forward
            t0 = time.perf_counter()
            optimizer.zero_grad()
            logits = model(input_ids)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, vocab_size),
                input_ids.reshape(-1)
            )
            if device == 'cuda':
                torch.cuda.synchronize()
            forward_times.append(time.perf_counter() - t0)

            # Backward
            t0 = time.perf_counter()
            loss.backward()
            if device == 'cuda':
                torch.cuda.synchronize()
            backward_times.append(time.perf_counter() - t0)

            # Optimizer
            t0 = time.perf_counter()
            optimizer.step()
            if device == 'cuda':
                torch.cuda.synchronize()
            optimizer_times.append(time.perf_counter() - t0)

            # Scheduler
            t0 = time.perf_counter()
            scheduler.step()
            if device == 'cuda':
                torch.cuda.synchronize()
            scheduler_times.append(time.perf_counter() - t0)

        return {
            'forward': sum(forward_times) / len(forward_times),
            'backward': sum(backward_times) / len(backward_times),
            'optimizer': sum(optimizer_times) / len(optimizer_times),
            'scheduler': sum(scheduler_times) / len(scheduler_times),
        }

    # Run benchmarks
    print("\n[1/2] Benchmarking PyTorch implementations...")
    pytorch_results = run_benchmark(use_custom_optimizer=False, num_iterations=100)

    print("\n[2/2] Benchmarking Custom implementations...")
    custom_results = run_benchmark(use_custom_optimizer=True, num_iterations=100)

    # Print results
    print("\n" + "=" * 80)
    print("RESULTS (ms per iteration)")
    print("=" * 80)
    print(f"\n{'Component':<20} {'PyTorch':>15} {'Custom':>15} {'Slowdown':>15}")
    print("-" * 80)

    for key in ['forward', 'backward', 'optimizer', 'scheduler']:
        pytorch_time = pytorch_results[key] * 1000
        custom_time = custom_results[key] * 1000
        ratio = custom_time / pytorch_time
        print(f"{key.capitalize():<20} {pytorch_time:>15.4f} {custom_time:>15.4f} {ratio:>15.2f}x")

    pytorch_total = sum(pytorch_results.values()) * 1000
    custom_total = sum(custom_results.values()) * 1000
    total_ratio = custom_total / pytorch_total
    print("-" * 80)
    print(f"{'Total':<20} {pytorch_total:>15.4f} {custom_total:>15.4f} {total_ratio:>15.2f}x")
    print("=" * 80)

    # Overhead analysis
    opt_overhead = (custom_results['optimizer'] - pytorch_results['optimizer']) * 1000
    sched_overhead = (custom_results['scheduler'] - pytorch_results['scheduler']) * 1000
    print(f"\nOptimizer overhead: {opt_overhead:+.4f} ms")
    print(f"Scheduler overhead: {sched_overhead:+.4f} ms")
    print(f"Total overhead: {opt_overhead + sched_overhead:+.4f} ms")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'benchmark':
        benchmark_optimizer_performance()
    elif len(sys.argv) > 1 and sys.argv[1] == 'real':
        benchmark_with_real_model()
    else:
        test_optimizer()
