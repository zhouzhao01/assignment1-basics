import torch
import torch.nn as nn
import math

class Linear(nn.Module):
    def __init__(self, d_in, d_out, device=None, dtype=None):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out

        self.W = nn.Parameter(
            data = torch.empty(d_out,d_in,device=device,dtype=dtype),
                                   requires_grad=True)
        
        torch.nn.init.trunc_normal_(self.W, mean=0.0, std=1.0/math.sqrt(d_in))
       

    def forward(self, in_features:torch.Tensor) -> torch.Tensor:
        return torch.einsum("... i, ... o i -> ... o", in_features, self.W)
    
class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()

        self.weights = nn.Parameter(
            data= torch.empty(num_embeddings,embedding_dim,device=device,dtype=dtype),
            requires_grad=True
        )

        nn.init.trunc_normal_(self.weights, mean=0.0, std=1.0/math.sqrt(embedding_dim))

    def forward(self,tokens_ids: torch.Tensor) -> torch.Tensor:
        return self.weights[tokens_ids]

class rmsnorm(nn.Module):
    def __init__(self, d_model:int, eps:float=1e-5, device=None, dtype=None):
        super().__init__()

        self.d_model = d_model
        self.eps = eps

        self.g = nn.Parameter(
            data = torch.empty(d_model,device=device,dtype=torch.float32),
            requires_grad=True
        )
        torch.nn.init.ones_(self.g)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype

        x = x.to(torch.float32)

        RMS = torch.sqrt( (x ** 2).sum(dim=-1) / self.d_model + self.eps)
        RMS_norm = x / RMS.unsqueeze(-1) * self.g

        return RMS_norm.to(in_dtype)

class SiLU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x:torch.Tensor):
        return x * torch.sigmoid(x)

class GLU(nn.Module):
    def __init__(self, dim_in:int, dim_out:int):
        super().__init__()

        self.linear_1 = Linear(dim_out, dim_in)
        self.linear_2 = Linear(dim_out, dim_in)
    
    def forward(self, x:torch.Tensor):
        return torch.sigmoid(self.linear_1(x)) * self.linear_2(x)


class SwiGLU(nn.Module):
    def __init__(self, dim_model:int, dff:int = None, device:str=None):
        super().__init__()

        self.dim_model = dim_model
        self.device = device

        if dff == None:
            self.dff = int(round(self.dim_model * 8 / 3 / 64) * 64)
        else:
            self.dff = dff

        self.linear_1 = nn.Parameter(torch.empty(self.dff,        self.dim_model,device=device))
        self.linear_2 = nn.Parameter(torch.empty(self.dim_model,  self.dff      ,device=device))
        self.linear_3 = nn.Parameter(torch.empty(self.dff,        self.dim_model,device=device))

        torch.nn.init.trunc_normal_(self.linear_1, mean=0.0, std=1.0/math.sqrt(self.dim_model))
        torch.nn.init.trunc_normal_(self.linear_2, mean=0.0, std=1.0/math.sqrt(self.dff))
        torch.nn.init.trunc_normal_(self.linear_3, mean=0.0, std=1.0/math.sqrt(self.dim_model))

        self.SiLU = SiLU()

    def forward(self, x:torch.Tensor):
        x1 = self.SiLU(torch.einsum("f d,... s d ->... s f",self.linear_1,x))
        x3 = torch.einsum("f d,... s d ->... s f",self.linear_3,x)
        output = torch.einsum("d f,... s f ->... s d",self.linear_2, x1 * x3)
        return output

class RoPE_fast(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        inv_freq = 1. / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_dim=1):
        seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()[:, None, None, :]
            self.sin_cached = emb.sin()[:, None, None, :]
        return self.cos_cached, self.sin_cached

# rotary pos emb helpers:
def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=x1.ndim - 1) # dim=-1 triggers a bug in torch < 1.8.0

@torch.jit.script
def apply_rotary_pos_emb(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

@torch.jit.script
def apply_rotary_pos_emb_single(q, cos, sin):
    return (q * cos) + (rotate_half(q) * sin)


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

class softmax(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x:torch.Tensor, dim:int) -> torch.Tensor:
        max_norm = x - x.max(dim=dim, keepdim=True).values
        numerator = torch.exp(max_norm)
        dominator = torch.sum(numerator,dim=dim, keepdim=True)

        return numerator / dominator
    
class scaled_dot_product_attention(nn.Module):
    def __init__(self, d_k:int, d_v:int, inf=torch.inf, device=None):
        super().__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.softmax_layer = softmax()
        self.inf = inf
        self.device = device
        self.scale = 1.0 / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    
    def forward(self, Q:torch.Tensor, K:torch.Tensor, V:torch.Tensor, mask=None):
        dot_product = torch.einsum("... c k, ... v k -> ... c v", Q, K) * self.scale

        if mask is not None:
            dot_product = dot_product.masked_fill(~mask.to(torch.bool), float('-inf'))
            # B  = torch.zeros(mask.shape, device=self.device)
            # B[~mask.to(torch.bool)] = self.inf
            # dot_product = dot_product - B
        
        return self.softmax_layer(dot_product, dim=-1) @ V

class multihead_self_attention(nn.Module):
    def __init__(self, d_model:int, num_heads:int, max_seq_len=None, theta=None, device=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.device = device 

        self.d_k = int(d_model / num_heads)
        self.d_v = self.d_k

        self.W_Q = nn.Parameter(torch.empty(num_heads * self.d_k, d_model, device=device))
        self.W_K = nn.Parameter(torch.empty(num_heads * self.d_k, d_model, device=device))
        self.W_V = nn.Parameter(torch.empty(num_heads * self.d_v, d_model, device=device))
        self.W_O = nn.Parameter(torch.empty(d_model, num_heads * self.d_v, device=device))

        torch.nn.init.trunc_normal_(self.W_Q, mean=0.0, std=1.0/math.sqrt(d_model))
        torch.nn.init.trunc_normal_(self.W_K, mean=0.0, std=1.0/math.sqrt(d_model))
        torch.nn.init.trunc_normal_(self.W_V, mean=0.0, std=1.0/math.sqrt(d_model))
        torch.nn.init.trunc_normal_(self.W_O, mean=0.0, std=1.0/math.sqrt(num_heads * self.d_v))

        self.SDPA = scaled_dot_product_attention(self.d_k,self.d_v, device=device)
        
        self.max_seq_len = max_seq_len
        self.theta = theta
        if theta and max_seq_len:
            self.RoPE = RoPE(theta=theta,d_k=self.d_k,max_seq_len=max_seq_len,device=device)

    def forward(self,  in_features:torch.Tensor, flag_mask=True):
        """
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.
        """
        batch_size = in_features.shape[0]
        d_in = in_features.shape[-1]
        sequence_length = in_features.shape[-2]

        # Check for NaN/Inf in input features
        if torch.isnan(in_features).any():
            raise ValueError(f"MHSA: NaN detected in in_features! Shape: {in_features.shape}")
        if torch.isinf(in_features).any():
            raise ValueError(f"MHSA: Inf detected in in_features! Shape: {in_features.shape}")

        if self.max_seq_len and (sequence_length > self.max_seq_len):
            print(f"  sequence_length: {sequence_length}")
            print(f"  in_features.shape: {in_features.shape}")
            print(f"  Expected: {self.max_seq_len}")
            raise ValueError(f"Unexpected sequence_length: {sequence_length}")


        W_Q_in = torch.einsum("k d, ... s d -> ... s k", self.W_Q, in_features)
        W_K_in = torch.einsum("k d, ... s d -> ... s k", self.W_K, in_features)
        W_V_in = torch.einsum("v d, ... s d -> ... s v", self.W_V, in_features)

        # Check for NaN/Inf after projections
        if torch.isnan(W_Q_in).any() or torch.isinf(W_Q_in).any():
            raise ValueError(f"MHSA: NaN/Inf in W_Q_in after projection!")
        if torch.isnan(W_K_in).any() or torch.isinf(W_K_in).any():
            raise ValueError(f"MHSA: NaN/Inf in W_K_in after projection!")

        # Apply RoPE to Query and Key
        if self.theta:
            for head in range(self.num_heads):
                W_Q_in[...,head * self.d_k: (head + 1) * self.d_k] = self.RoPE(W_Q_in[...,head * self.d_k: (head + 1) * self.d_k], torch.arange(0,sequence_length,1))
                W_K_in[...,head * self.d_k: (head + 1) * self.d_k] = self.RoPE(W_K_in[...,head * self.d_k: (head + 1) * self.d_k], torch.arange(0,sequence_length,1))

        # Reshape: [..., seq_len, num_heads * d_k] -> [..., seq_len, num_heads, d_k]
        Q = W_Q_in.view(batch_size, sequence_length, self.num_heads, self.d_k)
        K = W_K_in.view(batch_size, sequence_length, self.num_heads, self.d_k)
        V = W_V_in.view(batch_size, sequence_length, self.num_heads, self.d_v)
        # Transpose to: [..., num_heads, seq_len, d_k] for batched computation
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Causal Mask
        if flag_mask:
            mask = torch.tril(torch.ones(sequence_length, sequence_length, device=self.device)).unsqueeze(0).expand(batch_size, self.num_heads, -1, -1)
        else:
            mask = None
        
        attentions = self.SDPA(Q, K, V, mask)  # [..., num_heads, seq_len, d_k]
        attentions = attentions.transpose(1,2).contiguous()
        attentions = attentions.view(batch_size, sequence_length, self.num_heads * self.d_k)

        MHSA = torch.einsum("d v, ... s v -> ... s d", self.W_O, attentions)

        return MHSA


# PyTorch native MultiheadAttention wrapper (for performance comparison)
class multihead_self_attention_fast(nn.Module):
    """
    Fast implementation using PyTorch's nn.MultiheadAttention.
    Drop-in replacement for multihead_self_attention with same interface.
    Note: RoPE is NOT supported in this version.
    """
    def __init__(self, d_model:int, num_heads:int, max_seq_len=None, theta=None, device=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.device = device
        self.max_seq_len = max_seq_len

        # PyTorch's MultiheadAttention expects (seq_len, batch, d_model) by default
        # We'll use batch_first=True to get (batch, seq_len, d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=0.0,
            bias=False,
            batch_first=True,
            device=device
        )

        # For causal mask caching
        self.register_buffer('causal_mask', None)
        self._cached_seq_len = 0

    def _get_causal_mask(self, seq_len, device):
        """Get or create causal mask for given sequence length"""
        if self.causal_mask is None or self._cached_seq_len != seq_len:
            # PyTorch expects attn_mask of shape (seq_len, seq_len)
            # where True means "mask out" (ignore)
            mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
            self.register_buffer('causal_mask', mask)
            self._cached_seq_len = seq_len
        return self.causal_mask

    def forward(self, in_features:torch.Tensor, flag_RoPE=None, flag_mask=True):
        """
        Args:
            in_features: (batch, seq_len, d_model)
            flag_RoPE: Not supported in this fast version
            flag_mask: Whether to apply causal mask
        Returns:
            output: (batch, seq_len, d_model)
        """
        if flag_RoPE:
            raise NotImplementedError("RoPE is not supported in fast attention. Use original implementation.")

        batch_size, seq_len, d_model = in_features.shape

        if self.max_seq_len and seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max {self.max_seq_len}")

        # Get causal mask if needed
        attn_mask = self._get_causal_mask(seq_len, in_features.device) if flag_mask else None

        # PyTorch's MHA with batch_first=True
        # attn_output shape: (batch, seq_len, d_model)
        attn_output, _ = self.attn(
            in_features, in_features, in_features,
            attn_mask=attn_mask,
            need_weights=False,
            is_causal=flag_mask  # Use optimized causal attention if available
        )

        return attn_output


class transformer_block(nn.Module):
    def __init__(self, d_model:int, num_heads:int, d_ff:int, max_seq_len:int, theta:int, device, use_fast_attn=False):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta

        self.device = device
        self.use_fast_attn = use_fast_attn

        self.rmsnorm_1 = rmsnorm(d_model=d_model, device=device)

        # Choose attention implementation
        if use_fast_attn:
            self.MHSA = multihead_self_attention_fast(d_model=d_model, num_heads=num_heads,
                                                      max_seq_len=max_seq_len, theta=theta, device=device)
        else:
            self.MHSA = multihead_self_attention(d_model=d_model, num_heads=num_heads,
                                                 max_seq_len=max_seq_len, theta=theta, device=device)

        self.rmsnorm_2 = rmsnorm(d_model=d_model, device=device)
        self.feedforward = SwiGLU(dim_model=d_model,dff=d_ff, device=device)


    def forward(self, x:torch.Tensor, flag_mask=True) -> torch.Tensor :
        y = x + self.MHSA(self.rmsnorm_1(x), flag_mask=flag_mask)
        z = y + self.feedforward(self.rmsnorm_2(y))
        return z

class transformer_lm(nn.Module):
    "uv run pytest -k test_transformer_lm"
    def __init__(self,
                vocab_size:int,
                context_length:int,
                num_layers:int,
                d_model: int,
                num_heads: int,
                d_ff: int,
                rope_theta: float,
                device = None,
                use_fast_attn = False
                 ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.max_seq_len = context_length

        self.num_layers = num_layers

        self.d_models = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta

        self.device = device

        # should be vocab_size, embed_dim
        self.embedding_layer = Embedding(vocab_size, d_model, device=device)

        self.transformer_layers = nn.ModuleList(
            [transformer_block(d_model=d_model,
                               num_heads=num_heads,
                               d_ff=d_ff,
                               max_seq_len=context_length,
                               theta=rope_theta,
                               device=device,
                               use_fast_attn=use_fast_attn
                               )
                                for _ in range(num_layers)
                               ]) 
        
        self.rmsnorm = rmsnorm(d_model=d_model,device=device)
        self.lm_head = Linear(d_model, vocab_size,device=device)  # Output layer for language modeling
    
    def forward(self,in_indices:torch.Tensor) -> torch.Tensor:
        """
        in_indices: Int[Tensor, " batch_size sequence_length"],
        Float[Tensor, " batch_size sequence_length vocab_size"]:
        """

        embed = self.embedding_layer(in_indices) #-> [B, Seq_len, embed_dim]

        # Pass through all transformer layers
        x = embed
        for layer in self.transformer_layers:
            x = layer(x)

        x = self.rmsnorm(x)
        logits = self.lm_head(x)
        # Note: Usually don't apply softmax in forward pass - leave as logits
        return logits



# Unit test
def unit_RoPE_test():
    d_k = 4
    length_s = 6
    RoPE_layer = RoPE(theta=10000,d_k=d_k,max_seq_len=12)
    print(RoPE_layer.rotation_matrix)
    
    x = torch.randn(1,length_s,d_k)
    token_position = torch.arange(0,length_s,1)
    y = RoPE_layer(x,token_position)
    print(x.shape)
    print(y.shape)

def unit_softmax_test():
    softmax_cal = softmax()
    x = torch.rand(4,6)
    return softmax_cal(x,1)

def test_attention():
    """Simple test for scaled dot-product attention"""
    # Setup
    d_k, d_v = 64, 64
    batch_size, seq_len = 2, 10
    attention = scaled_dot_product_attention(d_k, d_v)
    
    # Test data
    Q = torch.randn(batch_size, seq_len, d_k)
    K = torch.randn(batch_size, seq_len, d_k)
    V = torch.randn(batch_size, seq_len, d_v)
    mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).expand(batch_size, -1, -1)
    
    # Test without mask
    output1 = attention(Q, K, V)
    assert output1.shape == (batch_size, seq_len, d_v)
    assert not torch.isnan(output1).any()
    
    # # Test with mask
    output2 = attention(Q, K, V, mask)
    assert output2.shape == (batch_size, seq_len, d_v)
    assert not torch.isnan(output2).any()
    
    print("All tests passed!")

def test_multihead_self_attention():
    """Minimal test for multihead_self_attention"""
    # Setup
    d_model = 128
    num_heads = 8
    batch_size = 2
    seq_len = 10

    # Create layer
    mhsa = multihead_self_attention(d_model, num_heads)

    # Test input
    x = torch.randn(batch_size, seq_len, d_model)

    # Forward pass
    output = mhsa(x)

    # Basic checks
    assert output.shape == (batch_size, seq_len, d_model), f"Shape mismatch: {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN"

    # Gradient test
    x_grad = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
    out_grad = mhsa(x_grad)
    loss = out_grad.sum()
    loss.backward()
    assert x_grad.grad is not None, "No gradients"

    print(f"MHSA test passed! Output shape: {output.shape}")

def test_transformer_block():
    """Minimal test for transformer_block"""
    # Setup
    d_model = 128
    num_heads = 8
    d_ff = 512
    max_seq_len = 32
    theta = 10000.0
    batch_size = 2
    seq_len = 16

    # Create block
    block = transformer_block(d_model, num_heads, d_ff, max_seq_len, theta)

    # Test input
    x = torch.randn(batch_size, seq_len, d_model)

    # Forward pass
    output = block(x)

    # Basic checks
    assert output.shape == x.shape, f"Shape mismatch: {output.shape}"
    assert torch.isfinite(output).all(), "Output contains NaN/inf"
    assert not torch.allclose(output, torch.zeros_like(output)), "Output is zero"

def test_transformer_lm():
    """Minimal test for transformer_lm - under 40 lines total"""
    # Model hyperparameters
    vocab_size = 1024
    context_length = 64
    num_layers = 2
    d_model = 128
    num_heads = 4
    d_ff = 256
    rope_theta = 10000.0

    # Create model
    model = transformer_lm(
        vocab_size=vocab_size,
        context_length=context_length,
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta
    )

    # Test input: batch_size=2, sequence_length=16
    batch_size, seq_len = 2, 27
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Forward pass
    logits = model(input_ids)

    # Verify output shape and values
    expected_shape = (batch_size, seq_len, vocab_size)
    assert logits.shape == expected_shape, f"Expected {expected_shape}, got {logits.shape}"
    assert torch.isfinite(logits).all(), "Model output contains NaN/inf"
    assert not torch.allclose(logits, torch.zeros_like(logits)), "Model output is all zeros"

def calculate_model_stats(model, vocab_size, context_length, num_layers, d_model, num_heads, d_ff):
    """Calculate parameter count and FLOPs for transformer_lm model"""

    print("=" * 60)
    print("TRANSFORMER_LM MODEL ANALYSIS")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Vocab size: {vocab_size:,}")
    print(f"  Context length: {context_length:,}")
    print(f"  Layers: {num_layers}")
    print(f"  Model dimension: {d_model}")
    print(f"  Attention heads: {num_heads}")
    print(f"  FFN dimension: {d_ff}")
    print()

    # Parameter count breakdown
    print("PARAMETER BREAKDOWN:")
    print("-" * 40)

    # 1. Token embeddings
    embed_params = vocab_size * d_model
    print(f"Token embeddings:     {embed_params:>12,} ({embed_params/1e6:.1f}M)")

    # 2. Per transformer block
    # Multi-head attention: W_Q, W_K, W_V each (d_model x d_model), W_O (d_model x d_model)
    attn_params_per_layer = 4 * d_model * d_model

    # RMSNorm: 2 per block (before attention, before FFN)
    norm_params_per_layer = 2 * d_model

    # SwiGLU FFN: linear_1 (d_ff x d_model), linear_2 (d_model x d_ff), linear_3 (d_ff x d_model)
    ffn_params_per_layer = (d_ff * d_model) + (d_model * d_ff) + (d_ff * d_model)

    total_per_layer = attn_params_per_layer + norm_params_per_layer + ffn_params_per_layer
    transformer_params = num_layers * total_per_layer

    print(f"Per layer:")
    print(f"  Attention:          {attn_params_per_layer:>12,} ({attn_params_per_layer/1e6:.1f}M)")
    print(f"  RMSNorm (x2):       {norm_params_per_layer:>12,} ({norm_params_per_layer/1e3:.1f}K)")
    print(f"  SwiGLU FFN:         {ffn_params_per_layer:>12,} ({ffn_params_per_layer/1e6:.1f}M)")
    print(f"  Total per layer:    {total_per_layer:>12,} ({total_per_layer/1e6:.1f}M)")
    print(f"All {num_layers} layers:        {transformer_params:>12,} ({transformer_params/1e6:.1f}M)")

    # 3. Final components
    final_norm_params = d_model
    lm_head_params = d_model * vocab_size

    print(f"Final RMSNorm:        {final_norm_params:>12,} ({final_norm_params/1e3:.1f}K)")
    print(f"LM head:              {lm_head_params:>12,} ({lm_head_params/1e6:.1f}M)")

    # Total parameters
    total_params = embed_params + transformer_params + final_norm_params + lm_head_params
    print("-" * 40)
    print(f"TOTAL PARAMETERS:     {total_params:>12,} ({total_params/1e6:.1f}M)")
    if total_params >= 1e9:
        print(f"                      {' ':>12} ({total_params/1e9:.2f}B)")

    # Verify with PyTorch's parameter count
    pytorch_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print()
    print("PYTORCH VERIFICATION:")
    print(f"PyTorch total:        {pytorch_params:>12,} ({pytorch_params/1e6:.1f}M)")
    print(f"Trainable params:     {trainable_params:>12,} ({trainable_params/1e6:.1f}M)")
    print(f"Match calculation:    {'✓' if pytorch_params == total_params else '✗'}")

    print()
    print("FLOP ANALYSIS (Forward Pass):")
    print("-" * 40)

    # FLOP calculation for different sequence lengths
    seq_lengths = [128, 256, 512, 1024]

    for seq_len in seq_lengths:
        if seq_len > context_length:
            continue

        print(f"Sequence length: {seq_len}")

        # 1. Embedding lookup: negligible FLOPs
        embed_flops = 0

        # 2. Per transformer layer FLOPs
        # Attention:
        # - QKV projections: 3 * seq_len * d_model * d_model
        # - Attention scores: seq_len * seq_len * d_model (for all heads)
        # - Attention weights * values: seq_len * seq_len * d_model
        # - Output projection: seq_len * d_model * d_model
        attn_flops_per_layer = (
            3 * seq_len * d_model * d_model +  # QKV projections
            seq_len * seq_len * d_model +      # Attention computation
            seq_len * seq_len * d_model +      # Apply attention to values
            seq_len * d_model * d_model        # Output projection
        )

        # FFN: 3 linear layers in SwiGLU
        # linear_1: seq_len * d_model * d_ff
        # linear_2: seq_len * d_ff * d_model
        # linear_3: seq_len * d_model * d_ff
        ffn_flops_per_layer = seq_len * d_model * d_ff * 3

        # RMSNorm: approximate as seq_len * d_model per norm (2 per layer)
        norm_flops_per_layer = 2 * seq_len * d_model

        layer_flops = attn_flops_per_layer + ffn_flops_per_layer + norm_flops_per_layer

        # 3. Final components
        final_norm_flops = seq_len * d_model
        lm_head_flops = seq_len * d_model * vocab_size

        total_flops = (
            embed_flops +
            num_layers * layer_flops +
            final_norm_flops +
            lm_head_flops
        )

        print(f"  Per layer:          {layer_flops/1e9:>8.2f} GFLOP")
        print(f"  All layers:         {(num_layers * layer_flops)/1e9:>8.2f} GFLOP")
        print(f"  LM head:            {lm_head_flops/1e9:>8.2f} GFLOP")
        print(f"  Total:              {total_flops/1e9:>8.2f} GFLOP")
        print()

    # Memory estimation (rough)
    print("MEMORY ESTIMATION (FP32):")
    print("-" * 40)
    param_memory_gb = total_params * 4 / (1024**3)  # 4 bytes per float32
    print(f"Parameters:           {param_memory_gb:>8.2f} GB")

    # Activation memory for max context length (very rough estimate)
    # Mainly attention scores and intermediate activations
    activation_memory_gb = (context_length**2 * num_heads * num_layers * 4) / (1024**3)
    print(f"Activations (est):    {activation_memory_gb:>8.2f} GB")
    print(f"Total (est):          {param_memory_gb + activation_memory_gb:>8.2f} GB")

    print("=" * 60)

if __name__ == "__main__":
    # unit_RoPE_test()
    # unit_softmax_test()
    # test_attention()
    # test_multihead_self_attention()
    # test_transformer_block()
    # test_transformer_lm()

    # Model configuration
    vocab_size = 32000
    context_length = 128
    num_layers = 48
    d_model = 512
    num_heads = 8
    d_ff = 2048

    # Create model
    model = transformer_lm(
        vocab_size=vocab_size,
        context_length=context_length,
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=10000.0
    )

    # Calculate and display model statistics
    calculate_model_stats(model, vocab_size, context_length, num_layers, d_model, num_heads, d_ff)