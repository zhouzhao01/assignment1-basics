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
            for p in group["params"]: # for each param in one param group

                # Step 1: get specific grad for specific param
                if p.grad is None:
                    continue
                grad = p.grad.data

                # Step 2: get Hyper-parameter from group["parameter"].
                # These hyper-parameter are created from __init__ defaults dic.
                beta_1, beta_2 = group['beta_1'], group['beta_2']
                alpha = group["alpha"]
                weight_decay = group["weight_decay"]

                # Step 3: Initialize State
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 1
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                # Step 4: Calculate 1st, 2nd Momentum
                # Compute adjusted learning rate
                
                state['exp_avg']     = beta_1 * state['exp_avg']    + (1 - beta_1) * grad
                state['exp_avg_sq']  = beta_2 * state['exp_avg_sq'] + (1 - beta_2) * grad * grad
                alpha_t = alpha * math.sqrt(1-beta_2**state['step']) / (1-beta_1**state['step'])

                # Step 5:
                # Update the parameter, using fancy modified gradients.
                p.data -= alpha_t * state['exp_avg'] / (torch.sqrt(state['exp_avg_sq']) + self.eps)
                # Apply Weight Decay.
                p.data -= p.data * alpha * weight_decay 

                state["step"] += 1 
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
        current_lr = self.get_lr(self.iteration)
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

if __name__ == "__main__":
    test_optimizer()
    

    