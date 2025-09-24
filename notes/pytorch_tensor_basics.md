# PyTorch Tensor Manipulation Guide

## Creating Tensors

```python
import torch

# From lists
a = torch.tensor([1, 2, 3])
b = torch.tensor([[1, 2], [3, 4]])

# Specific shapes
zeros = torch.zeros(3, 4)  # 3x4 matrix of zeros
ones = torch.ones(2, 3, 4)  # 2x3x4 tensor of ones
rand = torch.rand(3, 3)  # 3x3 random values [0, 1)
randn = torch.randn(3, 3)  # 3x3 normal distribution

# Like another tensor
x = torch.zeros_like(b)  # Same shape as b

# Ranges
arange = torch.arange(0, 10, 2)  # [0, 2, 4, 6, 8]
linspace = torch.linspace(0, 1, 5)  # 5 evenly spaced points
```

## Shape Operations

```python
# Get shape
x = torch.rand(2, 3, 4)
print(x.shape)  # torch.Size([2, 3, 4])
print(x.size())  # Same as shape
print(x.dim())  # 3 (number of dimensions)

# Reshape
y = x.view(6, 4)  # Reshape to 6x4 (must have same total elements)
y = x.reshape(6, 4)  # Similar to view but more flexible
y = x.flatten()  # Flatten to 1D

# Add/remove dimensions
y = x.unsqueeze(0)  # Add dimension at position 0
y = x.squeeze()  # Remove all dimensions of size 1
y = x.squeeze(1)  # Remove dimension 1 if size is 1

# Transpose
a = torch.rand(3, 4)
b = a.T  # Simple transpose
b = a.transpose(0, 1)  # Swap specific dimensions
b = a.permute(1, 0)  # Reorder dimensions
```

## Basic Arithmetic

```python
a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
b = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)

# Element-wise operations
c = a + b  # Addition
c = a - b  # Subtraction
c = a * b  # Element-wise multiplication
c = a / b  # Division
c = a ** 2  # Power

# Broadcasting (automatic dimension expansion)
a = torch.rand(3, 4)
b = torch.rand(4)  # Will broadcast to (3, 4)
c = a + b  # Works!

a = torch.rand(3, 1, 4)
b = torch.rand(1, 5, 4)
c = a + b  # Result shape: (3, 5, 4)
```

## Matrix Operations

```python
# Matrix multiplication
a = torch.rand(3, 4)
b = torch.rand(4, 5)

# Different ways to do matmul
c = torch.matmul(a, b)  # (3, 5)
c = a @ b  # Same as matmul
c = torch.mm(a, b)  # Only for 2D tensors

# Batch matrix multiplication
a = torch.rand(10, 3, 4)  # 10 matrices of 3x4
b = torch.rand(10, 4, 5)  # 10 matrices of 4x5
c = torch.bmm(a, b)  # (10, 3, 5)

# For higher dimensions
a = torch.rand(2, 3, 4, 5)
b = torch.rand(2, 3, 5, 6)
c = torch.matmul(a, b)  # (2, 3, 4, 6) - broadcasts over batch dims
```

## Einstein Summation (einsum) - VERY POWERFUL!

```python
# einsum notation: specify input dimensions and output dimensions
# Repeated indices are summed over

# Matrix multiplication
a = torch.rand(3, 4)
b = torch.rand(4, 5)
c = torch.einsum('ij,jk->ik', a, b)  # (3, 5)

# Batch matrix multiplication
a = torch.rand(10, 3, 4)
b = torch.rand(10, 4, 5)
c = torch.einsum('bij,bjk->bik', a, b)  # (10, 3, 5)

# Dot product
a = torch.rand(5)
b = torch.rand(5)
c = torch.einsum('i,i->', a, b)  # scalar

# Outer product
a = torch.rand(3)
b = torch.rand(4)
c = torch.einsum('i,j->ij', a, b)  # (3, 4)

# Transpose
a = torch.rand(3, 4)
b = torch.einsum('ij->ji', a)  # (4, 3)

# Batch transpose
a = torch.rand(10, 3, 4)
b = torch.einsum('bij->bji', a)  # (10, 4, 3)

# Diagonal
a = torch.rand(3, 3)
diag = torch.einsum('ii->i', a)  # Extract diagonal

# Trace (sum of diagonal)
trace = torch.einsum('ii->', a)  # scalar

# Complex example: attention scores
Q = torch.rand(8, 10, 64)  # (batch, seq_len, d_k)
K = torch.rand(8, 10, 64)  # (batch, seq_len, d_k)
scores = torch.einsum('bqd,bkd->bqk', Q, K)  # (8, 10, 10)
```

## Reduction Operations

```python
x = torch.rand(3, 4)

# Sum
total = x.sum()  # Sum all elements
row_sums = x.sum(dim=1)  # Sum along dimension 1
col_sums = x.sum(dim=0)  # Sum along dimension 0
keep_dim = x.sum(dim=1, keepdim=True)  # Keep dimension (3, 1)

# Mean
mean_all = x.mean()
mean_rows = x.mean(dim=1)

# Max/Min
max_val = x.max()  # Just the value
max_val, max_idx = x.max(dim=1)  # Values and indices
min_val = x.min()

# Along dimension with keepdim
max_vals = x.max(dim=1, keepdim=True).values  # (3, 1)

# Argmax/Argmin (indices only)
idx = x.argmax()  # Flattened index
idx = x.argmax(dim=1)  # Index along dimension

# Product
prod = x.prod()
prod_rows = x.prod(dim=1)
```

## Advanced Indexing and Slicing

```python
x = torch.rand(5, 6)

# Basic slicing (like NumPy)
a = x[0]  # First row
a = x[:, 0]  # First column
a = x[1:3, 2:5]  # Submatrix

# Fancy indexing
indices = torch.tensor([0, 2, 4])
a = x[indices]  # Select rows 0, 2, 4

# Boolean masking
mask = x > 0.5
a = x[mask]  # 1D tensor of elements > 0.5

# Gather (select from dimension)
x = torch.rand(3, 4)
idx = torch.tensor([[0, 1], [2, 3], [1, 2]])  # (3, 2)
result = torch.gather(x, dim=1, index=idx)  # (3, 2)

# Index select
rows = torch.index_select(x, dim=0, index=torch.tensor([0, 2]))
```

## Useful Operations for ML

```python
# Clamp (clip values)
x = torch.rand(3, 4)
clamped = torch.clamp(x, min=0.2, max=0.8)

# Where (conditional selection)
a = torch.rand(3, 4)
b = torch.rand(3, 4)
mask = a > 0.5
result = torch.where(mask, a, b)  # Take from a if mask True, else b

# Concatenate
a = torch.rand(3, 4)
b = torch.rand(2, 4)
c = torch.cat([a, b], dim=0)  # (5, 4)

# Stack (add new dimension)
a = torch.rand(3, 4)
b = torch.rand(3, 4)
c = torch.stack([a, b], dim=0)  # (2, 3, 4)

# Split
x = torch.rand(10, 4)
splits = torch.split(x, 3, dim=0)  # Split into chunks of 3

# Expand (broadcast to larger size)
x = torch.rand(1, 4)
y = x.expand(3, 4)  # Broadcast to (3, 4)

# Repeat
x = torch.rand(2, 3)
y = x.repeat(2, 3)  # (4, 9) - repeat 2x along dim0, 3x along dim1
```

## Special Functions for ML

```python
# Exponential and logarithm
x = torch.rand(3, 4)
exp_x = torch.exp(x)
log_x = torch.log(x)

# Activation functions
sigmoid = torch.sigmoid(x)
tanh = torch.tanh(x)
relu = torch.relu(x)

# For softmax implementation
exp_x = torch.exp(x - x.max(dim=1, keepdim=True).values)  # Numerical stability
softmax = exp_x / exp_x.sum(dim=1, keepdim=True)

# For cross-entropy
log_softmax = torch.log_softmax(x, dim=1)

# Norm
l2_norm = torch.norm(x, p=2, dim=1)  # L2 norm along dimension
```

## Tips for Your Implementations

1. **Linear Layer**: You'll need `einsum` or `matmul` for matrix multiplication
2. **Embedding**: Use indexing to select rows from weight matrix
3. **Attention**: `einsum` is perfect for QK^T computation
4. **Softmax**: Remember numerical stability (subtract max before exp)
5. **RMSNorm**: Use `mean`, `sqrt`, and element-wise operations
6. **SiLU**: x * sigmoid(x)
7. **Cross-Entropy**: Combine log_softmax and negative log likelihood

## Common Patterns

```python
# Broadcasting for batch operations
batch_size, seq_len, d_model = 32, 100, 512
x = torch.rand(batch_size, seq_len, d_model)
weights = torch.rand(d_model, d_model)

# Apply same weights to all batches
output = torch.einsum('bsd,dk->bsk', x, weights)

# Keeping dimensions for broadcasting
x = torch.rand(32, 100, 512)
mean = x.mean(dim=-1, keepdim=True)  # (32, 100, 1)
normalized = x - mean  # Broadcasting happens automatically

# Masking pattern
scores = torch.rand(32, 100, 100)
mask = torch.triu(torch.ones(100, 100), diagonal=1).bool()
scores.masked_fill_(mask, -float('inf'))
```

## Debugging Tips

```python
# Check shapes frequently
print(f"Shape: {tensor.shape}")
print(f"Device: {tensor.device}")
print(f"Dtype: {tensor.dtype}")

# Use assertions
assert tensor.shape == (expected_shape)

# Check for NaN/Inf
has_nan = torch.isnan(tensor).any()
has_inf = torch.isinf(tensor).any()
```

Remember: PyTorch operations are similar to NumPy but:
- Use `torch` functions instead of `np`
- GPU support with `.cuda()` or `.to(device)`
- Automatic differentiation with `requires_grad=True`
- Broadcasting rules are the same as NumPy