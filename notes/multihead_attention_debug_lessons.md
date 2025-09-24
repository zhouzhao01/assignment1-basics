# Multihead Self-Attention Debugging Lessons

## Issue Summary
The multihead self-attention implementation was failing tests due to:
1. Incorrect weight loading in the adapter
2. Missing causal masking in the forward pass

## Key Debugging Steps

### 1. Weight Loading Issue
**Problem**: In `tests/adapters.py`, weights weren't being loaded correctly:
```python
# Wrong - this overwrites the load_state_dict method!
MHSA.load_state_dict = {
    "W_Q": q_proj_weight,
    ...
}
```

**Solution**: Call the method properly:
```python
# Correct
MHSA.load_state_dict({
    "W_Q": q_proj_weight,
    "W_K": k_proj_weight,
    "W_V": v_proj_weight,
    "W_O": o_proj_weight
})
```

### 2. Causal Masking Requirement
**Problem**: Test expected causal masking but it wasn't being applied by default.

**Solution**: Added `flag_mask=True` when calling the model in the adapter.

## Implementation Insights

### Correct Multi-Head Processing
The implementation correctly:
1. Projects input to Q, K, V using batched matrix multiplication
2. Splits attention computation per head (loop through heads)
3. Concatenates head outputs using `torch.cat(attentions, dim=-1)`
4. Applies output projection

### Masking Logic
The masking approach `dot_product - B` where `B[~mask] = inf` is correct:
- Subtracting infinity gives `-inf` for masked positions
- Softmax converts `-inf` to 0, effectively masking those positions

## Lessons Learned

### 1. Read Test Expectations Carefully
- Tests may have implicit assumptions (e.g., causal masking for transformers)
- Even when not explicitly documented, consider typical use cases

### 2. Debug Systematically
- Start with obvious issues (weight loading)
- Isolate components (test SDPA separately)
- Compare configurations (with/without mask)
- Verify actual code being executed

### 3. Value Magnitudes Are Diagnostic
- Wildly wrong values (~200) → fundamental issue (weights not loaded)
- Close but not exact (~0.08 vs ~0.24) → missing feature (masking)

### 4. Default Parameters Matter
- Consider what defaults make sense for typical use cases
- For transformer self-attention, causal masking is often expected

### 5. Python Gotchas
- `obj.method = value` overwrites the method, doesn't call it
- Always verify tensor shapes at key points
- Check device/dtype compatibility when creating tensors

## Why Causal Masking?
Multi-head self-attention in transformers typically uses causal masking to prevent the model from "looking ahead" at future tokens during training. This is standard for autoregressive language models where each token should only attend to previous tokens.

## Final Working Configuration
- Weights loaded correctly via `load_state_dict()`
- Causal masking applied with `flag_mask=True`
- Head-wise attention computation with proper concatenation
- All tests passing