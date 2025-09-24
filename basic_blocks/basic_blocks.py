import torch
import torch.nn as nn

from einops import einsum

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(
            data = torch.empty(out_features,in_features,device=device,dtype=dtype),
                                   requires_grad=True)
        
        torch.nn.init.trunc_normal_(self.W)
       

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return x @ self.W.T
    
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
        max_norm = x - x.max(dim=dim, keepdim=True)
        numerator = torch.exp(max_norm)
        dominator = torch.sum(numerator,dim=dim, keepdim=True)

        return numerator / dominator

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

if __name__ == "__main__":
    # unit_RoPE_test()
    unit_softmax_test()