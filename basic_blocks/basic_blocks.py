import torch
import torch.nn as nn

from einops import einsum

class Linear(nn.Module):
    def __init__(self, d_in, d_out, device=None, dtype=None):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out

        self.W = nn.Parameter(
            data = torch.empty(d_out,d_in,device=device,dtype=dtype),
                                   requires_grad=True)
        
        torch.nn.init.trunc_normal_(self.W)
       

    def forward(self, in_features:torch.Tensor) -> torch.Tensor:
        return torch.einsum("... i, o i -> ... o", in_features, self.W)
    
class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()

        self.weights = nn.Parameter(
            data= torch.empty(num_embeddings,embedding_dim,device=device,dtype=dtype),
            requires_grad=True
        )

        nn.init.trunc_normal_(self.weights)

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
    def __init__(self, dim_model:int, dff:int = None):
        super().__init__()

        self.dim_model = dim_model

        if dff == None:
            self.dff = int(round(self.dim_model * 8 / 3 / 64) * 64)
        else:
            self.dff = dff

        self.linear_1 = Linear(self.dff,self.dim_model)
        self.linear_2 = Linear(self.dim_model,self.dff)
        self.linear_3 = Linear(self.dff,self.dim_model)

        self.SiLU = SiLU()

    def forward(self, x:torch.Tensor):
        return self.linear_2(
            self.SiLU(
                self.linear_1(x)) * (self.linear_3(x)
                )
            )

class RoPE(nn.Module):
    def __init__(self, theta:float, d_k:int, max_seq_len:int, device=None):
        super().__init__()

        self.theta = theta
        self.d_k = d_k

        # TODO: OPTIMIZATION STEP 1 - Replace rotation matrices with sin/cos caches
        # Current approach: O(max_seq_len × d_k²) memory for full matrices
        self.rotation_matrix = torch.zeros(max_seq_len,d_k,d_k,device=device)
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
        rotation_matrix = torch.zeros(self.d_k, self.d_k)

        for k in torch.arange(1, self.d_k/2+1, 1):
            theta_i_d =  token_position / (self.theta ** ((2 * (k - 1)) / self.d_k))
            rotation_subblock_element = torch.tensor(
                [
                    [torch.cos(theta_i_d), -torch.sin(theta_i_d)],
                    [torch.sin(theta_i_d),  torch.cos(theta_i_d)]
                ],
                dtype=torch.float32
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
        for position in token_position:
            """
            Be careful. It should be x * R.T.
            """
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
    def __init__(self, d_k:int, d_v:int, inf=torch.inf):
        super().__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.softmax_layer = softmax()
        self.inf = inf
    
    def forward(self, Q:torch.Tensor, K:torch.Tensor, V:torch.Tensor, mask=None):
        dot_product = torch.einsum("... c k, ... v k -> ... c v", Q, K)
        if mask != None:
            B  = torch.zeros(mask.shape)
            B[~mask.to(torch.bool)] = self.inf
            dot_product = dot_product - B
        
        return self.softmax_layer(dot_product / torch.sqrt(torch.tensor(self.d_k)),dim=-1) @ V

class multihead_self_attention(nn.Module):
    def __init__(self, d_model:int, num_heads:int, max_seq_len=None, theta=None, device=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        self.d_k = int(d_model / num_heads)
        self.d_v = self.d_k

        self.W_Q = nn.Parameter(torch.empty(num_heads * self.d_k, d_model),requires_grad=True)
        self.W_K = nn.Parameter(torch.empty(num_heads * self.d_k, d_model),requires_grad=True)
        self.W_V = nn.Parameter(torch.empty(num_heads * self.d_v, d_model),requires_grad=True)
        self.W_O = nn.Parameter(torch.empty(d_model, num_heads * self.d_v),requires_grad=True)

        torch.nn.init.trunc_normal_(self.W_Q)
        torch.nn.init.trunc_normal_(self.W_K)
        torch.nn.init.trunc_normal_(self.W_V)
        torch.nn.init.trunc_normal_(self.W_O)

        self.SDPA = scaled_dot_product_attention(self.d_k,self.d_v)

        if theta and max_seq_len:
            self.RoPE = RoPE(theta=theta,d_k=self.d_k,max_seq_len=max_seq_len,device=device)

    def forward(self,  in_features:torch.Tensor, flag_RoPE=None, flag_mask=True):
        """
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.
        """
        batch_size = in_features.shape[0]
        d_in = in_features.shape[-1] 
        sequence_length = in_features.shape[-2] 

        W_Q_in = torch.einsum("k d, ... s d -> ... s k", self.W_Q, in_features)
        W_K_in = torch.einsum("k d, ... s d -> ... s k", self.W_K, in_features)
        W_V_in = torch.einsum("v d, ... s d -> ... s v", self.W_V, in_features)

        # Apply RoPE to Query and Key
        if flag_RoPE:
            for head in range(self.num_heads):
                W_Q_in[...,head * self.d_k: (head + 1) * self.d_k] = self.RoPE(W_Q_in[...,head * self.d_k: (head + 1) * self.d_k], torch.arange(0,sequence_length,1))
                W_K_in[...,head * self.d_k: (head + 1) * self.d_k] = self.RoPE(W_K_in[...,head * self.d_k: (head + 1) * self.d_k], torch.arange(0,sequence_length,1))

        # Causal Mask
        if flag_mask:
            mask = torch.tril(torch.ones(sequence_length, sequence_length)).unsqueeze(0).expand(batch_size, -1, -1)
        else:
            mask = None
        
        attentions = []
        for head in range(self.num_heads):
            attention = self.SDPA( Q=W_Q_in[...,head * self.d_k: (head + 1) * self.d_k],
                                   K=W_K_in[...,head * self.d_k: (head + 1) * self.d_k], 
                                   V=W_V_in[...,head * self.d_k: (head + 1) * self.d_k],
                                   mask=mask) # [..., s v]
            attentions.append(attention)
        attentions = torch.cat(attentions, dim=-1)
        MHSA = torch.einsum("d v, ... s v -> ... s d", self.W_O, attentions)

        return MHSA
        

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

if __name__ == "__main__":
    # unit_RoPE_test()
    # unit_softmax_test()
    # test_attention()
    test_multihead_self_attention()