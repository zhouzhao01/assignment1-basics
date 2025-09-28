# Custom AdamW Optimizer for CS336.
# Zhao Zhou
# 2025.9.26

import torch
import torch.nn as nn

from  collections.abc import Callable, Iterable
from typing import Optional


import math

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, weight_decay=0.01, betas = (0.9, 0.999), eps=1e-8, flag_lr_schedule=False):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if betas[0] < 0:
            raise ValueError(f"Invalid betas[0]: {betas[0]}")
        if betas[1] < 0:
            raise ValueError(f"Invalid betas[1]: {betas}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")   
        
        self.eps = eps
        self.flag_lr_schedule = flag_lr_schedule
        if flag_lr_schedule is True:
            self.lr_schedule = lr_cosine_schedule(max_learning_rate=1, min_learning_rate=0, warmup_iters=1000, cosine_cycle_iters=1000)

        defaults = {
            "alpha": lr,
            "beta_1": betas[0],
            "beta_2": betas[1],
            "weight_decay": weight_decay,
        }

        super().__init__(params, defaults)

    def step(self, iterations:int, closure:Optional[Callable]=None):
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

                # Step 4: Calculate 1st, 2nd Moment
                # Compute adjusted learning rate
                
                state['exp_avg']     = beta_1 * state['exp_avg']    + (1 - beta_1) * grad
                state['exp_avg_sq']  = beta_2 * state['exp_avg_sq'] + (1 - beta_2) * grad * grad
                alpha_t = alpha * math.sqrt(1-beta_2**state['step']) / (1-beta_1**state['step'])

                # Step 5:
                # Update the parameter, using fancy modified gradients.
                p.data = p.data - alpha_t * state['exp_avg'] / (torch.sqrt(state['exp_avg_sq']) + self.eps)
                # Apply Weight Decay.
                if self.flag_lr_schedule == True:
                    p.data = p.data - alpha * weight_decay * p.data * self.lr_schedule(iterations)
                else:
                    p.data = p.data - alpha * weight_decay * p.data
                state["step"] += 1 
        return loss

class lr_cosine_schedule(nn.Module):
    def __init__(self, max_learning_rate:float, min_learning_rate:float, warmup_iters:int, cosine_cycle_iters:int):
        super().__init__()
        self.a_max = max_learning_rate
        self.a_min = min_learning_rate
        self.T_w = warmup_iters
        self.T_c = cosine_cycle_iters

    def forward(self, t:int):
        if t < 0:
            raise ValueError(f"Invalid learning rate step: {t}")
        elif t < self.T_w:
            lr = t / self.T_w * self.a_max
        elif t < self.T_c:
            lr = self.a_min + 0.5 * (1 + math.cos((t-self.T_w)/(self.T_c-self.T_w)*math.pi)) * (self.a_max - self.a_min)
        else:
            lr = self.a_min

        return lr

class grad_clip(nn.Module):
    def __init__(self, max_l2_norm:float, eps:float=1e-6):
        super().__init__()
        self.max_l2_norm = max_l2_norm
        self.eps = eps

    def forward(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

        Args:
            parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
            max_l2_norm (float): a positive value containing the maximum l2-norm.

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
