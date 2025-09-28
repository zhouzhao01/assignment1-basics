# Transformer Block Debugging: The Einsum Batch Dimension Bug

## Date: 2025-09-24
## Component: `transformer_block` and `SwiGLU`

## The Problem
Transformer block test was failing with large numerical differences:
- Expected values: ~0.06, ~1.07, ~-0.69
- Actual values: ~7.0, ~1.9, ~-1.4
- Max relative difference: 25870.203

## Root Cause: Missing Batch Dimension in Einsum

### The Bug
In `SwiGLU.forward()`, the einsum patterns were missing the `...` ellipsis:

```python
# WRONG - Missing batch dimension handling
def forward(self, x:torch.Tensor):
    x1 = self.SiLU(torch.einsum("f d, s d -> s f", self.linear_1, x))
    x3 = torch.einsum("f d, s d -> s f", self.linear_3, x)
    output = torch.einsum("d f, s f -> s d", self.linear_2, x1 * x3)
    return output
```

### The Fix
Added `...` to preserve batch dimensions:

```python
# CORRECT - Preserves batch dimension
def forward(self, x:torch.Tensor):
    x1 = self.SiLU(torch.einsum("f d,... s d ->... s f", self.linear_1, x))
    x3 = torch.einsum("f d,... s d ->... s f", self.linear_3, x)
    output = torch.einsum("d f,... s f ->... s d", self.linear_2, x1 * x3)
    return output
```

## What Happened Without `...`

### Input Shape Analysis
- Input `x`: `[batch, seq, d_model]` e.g., `[2, 16, 64]`
- Expected output: `[batch, seq, d_ff]` e.g., `[2, 16, 128]`

### Without `...` (Buggy Behavior)
```python
torch.einsum("f d, s d -> s f", linear_1, x)
```
- Einsum interpreted `x` as just `[s, d]` (last 2 dimensions)
- **Summed over the batch dimension** (collapsed it)
- Output shape: `[seq, d_ff]` instead of `[batch, seq, d_ff]`
- This caused incorrect aggregation across batch elements

### With `...` (Correct Behavior)
```python
torch.einsum("f d, ... s d -> ... s f", linear_1, x)
```
- The `...` means "keep all other dimensions unchanged"
- Properly preserves batch dimension
- Output shape: `[batch, seq, d_ff]` as expected

## Key Insights

1. **Einsum Default Behavior**: Without explicit specification, einsum sums over unmentioned dimensions
2. **The `...` Operator**: Crucial for preserving leading dimensions in batched operations
3. **Debugging Strategy**: When transformers produce wildly incorrect values, check dimension handling first

## Architecture Context

The transformer block uses pre-norm architecture:
```python
def forward(self, x, flag_RoPE=True, flag_mask=True):
    y = x + self.MHSA(self.rmsnorm_1(x), flag_RoPE=True, flag_mask=True)
    z = y + self.feedforward(self.rmsnorm_2(y))  # Not x!
    return z
```

Note: The second residual connection uses `y` (output of first block), not `x` (original input).

## Test Result
After fixing the einsum patterns:
- `uv run pytest -k test_transformer_block` ✅ PASSED
- All numerical values now match expected snapshots

## General Rule
**Always use `...` in einsum when working with batched tensors unless you explicitly want to reduce over leading dimensions.**

## Related Files
- `/basic_blocks/basic_blocks.py` - `SwiGLU` class
- `/tests/adapters.py` - `run_transformer_block()`
- Test command: `uv run pytest -k test_transformer_block`