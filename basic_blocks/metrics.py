import torch
import torch.nn as nn
from torch.nn import functional as F
from jaxtyping import Bool, Float, Int


class cross_entropy_loss(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self,
                inputs: Float[torch.Tensor, " ... vocab_size"], 
                targets: Int[torch.Tensor, " ..."]):
        
        return self.cal_cross_entropy_loss(inputs,targets)
    
    def cal_cross_entropy_loss(self,
                inputs: Float[torch.Tensor, " ... vocab_size"], 
                targets: Int[torch.Tensor, " ..."]): 
        vocab_size = inputs.shape[-1]
        
        # Normalize on the first batch size.
        inputs = inputs - torch.max(inputs,dim=-1,keepdim=True)[0] 
        
        # Organize to single batch
        inputs_flattened = inputs.view(-1, vocab_size) # -> ... vocab_size, 整理成单 sample.
        targets_flattened = targets.flatten() # 整理成单 sample.

        # Fancy Indexing
        batch_index = torch.arange(len(targets_flattened))
        numerator = inputs_flattened[batch_index, targets_flattened] # [batch * 2] :[batch_index colomn, targets_colomn]. Desired position probility
        dominator = torch.exp(inputs_flattened).sum(dim=-1) # scalar.

        log_prob = numerator - torch.log(dominator)

        return - log_prob.mean()


    
def unit_test_cross_entropy_loss():
    # Simple test case
    vocab_size = 1024
    batch_size = 2
    seq_len = 32

    inputs = torch.randn(batch_size, seq_len, vocab_size)
    inputs = inputs * 1e-6
    targets = torch.randint(low=0, high=vocab_size, size=[batch_size, seq_len])

    # Test our implementation
    loss_fn = cross_entropy_loss()
    our_loss = loss_fn(inputs, targets)
    expected_loss = torch.nn.functional.cross_entropy(inputs.view(batch_size*seq_len,-1), targets.flatten())

    # Compare with PyTorch's built-in
    # Assert they're close (within numerical precision)
    assert torch.allclose(our_loss, expected_loss, atol=1e-6)
    print(f"✅ Test passed! Our loss: {our_loss:.4f}, Expected: {expected_loss:.4f}")

if __name__ == "__main__":
    # Run the test
    # unit_test_cross_entropy_loss()
    pass



