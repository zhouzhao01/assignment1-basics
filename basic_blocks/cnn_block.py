"""
CNN Block Implementation Exercise for CS336
==========================================

PROBLEM DESCRIPTION:
Implement a 2D Convolutional Neural Network block from scratch without using torch.nn.Conv2d.
Your implementation should handle the core convolution operation using only basic tensor operations.

LEARNING OBJECTIVES:
1. Understand how convolution operations work at the tensor level
2. Implement padding, stride, and kernel operations manually
3. Handle batch processing and multiple input/output channels
4. Gain insight into memory layout and computational efficiency

IMPLEMENTATION REQUIREMENTS:
- Complete the custom_cnn2d class that inherits from nn.Module
- Support configurable kernel size, stride, padding, input channels, and output channels
- Initialize weights using appropriate initialization (e.g., Xavier/He initialization)
- Include bias terms (optional parameter)
- Handle arbitrary batch sizes and spatial dimensions

KEY CONCEPTS TO CONSIDER:
- How does the sliding window operation work?
- What's the relationship between input size, kernel size, stride, padding, and output size?
- How do you handle multiple input and output channels?
- What's the most efficient way to implement this without explicit loops?

MATHEMATICAL FOUNDATION:
For a 2D convolution with:
- Input: (batch_size, in_channels, height_in, width_in)
- Kernel: (out_channels, in_channels, kernel_height, kernel_width)
- Output: (batch_size, out_channels, height_out, width_out)

Output dimensions:
height_out = (height_in + 2*padding - kernel_height) // stride + 1
width_out = (width_in + 2*padding - kernel_width) // stride + 1

HINTS FOR IMPLEMENTATION:
1. Think about torch.nn.functional.unfold - what does it do and why is it useful?
2. Consider how you can reshape tensors to turn convolution into matrix multiplication
3. What's the difference between 'valid' and 'same' padding?
4. How does stride affect the sampling of your sliding window?

YOUR TASK:
Complete the custom_cnn2d class below and make sure it passes all the test cases.
Think through each step carefully and ask yourself "why" at each decision point.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class custom_cnn2d(nn.Module):
    """
    2D Convolution layer implemented from scratch.

    This implementation should NOT use torch.nn.Conv2d or torch.nn.functional.conv2d
    Instead, use basic tensor operations to implement the convolution.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, device=None, dtype=None):
        """
        Initialize the custom_cnn2d layer.

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (int or tuple): Size of the convolution kernel
            stride (int or tuple): Stride of the convolution
            padding (int or tuple): Padding added to input
            bias (bool): Whether to include bias term
            device: Device to place tensors on
            dtype: Data type for tensors

        QUESTIONS TO THINK ABOUT:
        - How should you initialize the weights?
        - What shape should the weight tensor be?
        - If bias is True, what shape should the bias tensor be?
        - How do you handle kernel_size/stride/padding as either int or tuple?
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if type(kernel_size) is int:
            self.kernel_element = kernel_size ** 2
            self.kernel_size = (kernel_size, kernel_size)
            
        else:
            self.kernal_size = kernel_size
            self.kernel_element = kernel_size[0] * kernel_size[1]

        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.device = device
        self.dtype = dtype

        self.weights = nn.ModuleList(
            [nn.Linear(in_features=self.kernel_element, 
                  out_features=1,
                  bias=bias,
                  device=device,
                  dtype=dtype
                  )
                  for _ in range(out_channels)]
                  )
        
        
        # TODO 初始化
        # TODO: Store parameters and create weight/bias tensors
        # Hint: Look at how other modules in basic_blocks.py handle initialization

    def forward(self, x):
        """
        Forward pass of the convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out)

        QUESTIONS TO THINK ABOUT:
        - How can you implement convolution without explicit nested loops?
        - What tensor operations can help you vectorize this computation?
        - How do you handle padding?
        - How do you handle stride?

        MAJOR HINT: Look into torch.nn.functional.unfold and think about how convolution
        can be reformulated as matrix multiplication. The unfold operation can extract
        sliding windows from your input, which you can then multiply with your kernels.
        """
        x = x.to(self.device)
        batch_size, in_channels, height, width = x.shape

        if self.padding > 0:
            x_pad = torch.zeros(batch_size, in_channels, height+ 2 * self.padding, width+ 2 * self.padding, device=self.device)
            x_pad[..., self.padding:self.padding+height ,self.padding:self.padding+width] = x
            _, _, height_pad, width_pad = x_pad.shape
        

        height_index_ls = torch.arange(0, height_pad - self.kernel_size[0] + 1, self.stride)
        width_index_ls  = torch.arange(0, width_pad - self.kernel_size[1] + 1,  self.stride)
        
        output = torch.zeros(
            (batch_size,self.out_channels,len(height_index_ls),len(width_index_ls)),
            device=self.device,
            dtype=self.dtype
        )

        for h_index, height_index in enumerate(height_index_ls):
            for w_index, width_index in enumerate(width_index_ls):

                x_pad_sliced = x_pad[...,height_index:height_index+self.kernel_size[0],width_index:width_index+self.kernel_size[1]]
                for out_channel_index in range(self.out_channels):
                    kernel_layer = self.weights[out_channel_index]
                    for in_channel_index in range(self.in_channels):
                        output[...,out_channel_index, h_index, w_index] = output[...,out_channel_index, h_index, w_index] +  kernel_layer(x_pad_sliced[...,in_channel_index,:,:].reshape([batch_size,self.kernel_element])).squeeze() / self.in_channels
                    
        return output

class custom_cnn2d(nn.Module):
    """
    2D Convolution layer implemented from scratch.

    This implementation should NOT use torch.nn.Conv2d or torch.nn.functional.conv2d
    Instead, use basic tensor operations to implement the convolution.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, device=None, dtype=None):
        """
        Initialize the custom_cnn2d layer.

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (int or tuple): Size of the convolution kernel
            stride (int or tuple): Stride of the convolution
            padding (int or tuple): Padding added to input
            bias (bool): Whether to include bias term
            device: Device to place tensors on
            dtype: Data type for tensors

        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if type(kernel_size) is int:
            self.kernel_element = kernel_size ** 2
            self.kernel_size = (kernel_size, kernel_size)
            
        else:
            self.kernel_size = kernel_size
            self.kernel_element = kernel_size[0] * kernel_size[1]

        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.device = device
        self.dtype = dtype

        self.weights = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size[0], kernel_size[1],device=device,dtype=dtype))
        self.bias = nn.Parameter(torch.empty(out_channels),device=device,dtype=dtype)
        
        # TODO 初始化
        # TODO: Store parameters and create weight/bias tensors
        # Hint: Look at how other modules in basic_blocks.py handle initialization

    def forward(self, x):
        """
        Forward pass of the convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out)
        """
        x = x.to(self.device)
        batch_size, in_channels, height, width = x.shape

        if self.padding > 0:
            x_pad = torch.zeros(batch_size, in_channels, height+ 2 * self.padding, width+ 2 * self.padding, device=self.device)
            x_pad[..., self.padding:self.padding+height ,self.padding:self.padding+width] = x
            _, _, height_pad, width_pad = x_pad.shape
        

        height_index_ls = torch.arange(0, height_pad - self.kernel_size[0] + 1, self.stride)
        width_index_ls  = torch.arange(0, width_pad - self.kernel_size[1] + 1,  self.stride)
        
        output = torch.zeros(
            (batch_size,self.out_channels,len(height_index_ls),len(width_index_ls)),
            device=self.device,
            dtype=self.dtype
        )



        for h_index, height_index in enumerate(height_index_ls):
            for w_index, width_index in enumerate(width_index_ls):

                x_pad_sliced = x_pad[...,height_index:height_index+self.kernel_size[0],width_index:width_index+self.kernel_size[1]]
                for out_channel_index in range(self.out_channels):
                    kernel_layer = self.weights[out_channel_index]
                    for in_channel_index in range(self.in_channels):
                        output[...,out_channel_index, h_index, w_index] = output[...,out_channel_index, h_index, w_index] +  kernel_layer(x_pad_sliced[...,in_channel_index,:,:].reshape([batch_size,self.kernel_element])).squeeze()
                    
        return output

def test_custom_cnn2d():
    """
    Simple test - let it crash if there are bugs for easier debugging.
    """
    print("Testing custom_cnn2d implementation...")

    # Basic test
    torch.manual_seed(42)
    x = torch.randn(2, 3, 8, 8)  # batch=2, channels=3, height=8, width=8

    # Create layers
    conv_custom = custom_cnn2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
    conv_pytorch = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)

    # Copy weights for comparison
    # with torch.no_grad():
    #     conv_pytorch.weight.copy_(conv_custom.weights)
    #     if conv_custom.bias is not None:
    #         conv_pytorch.bias.copy_(conv_custom.bias)

    # Forward pass
    output_custom = conv_custom(x)
    output_pytorch = conv_pytorch(x)

    print(f"Input shape: {x.shape}")
    print(f"Custom output: {output_custom.shape}")
    print(f"PyTorch output: {output_pytorch.shape}")
    print(f"Shapes match: {output_custom.shape == output_pytorch.shape}")
    print(f"Values close: {torch.allclose(output_custom, output_pytorch, atol=1e-6)}")

    # Quick gradient test
    x_grad = torch.randn(1, 2, 5, 5, requires_grad=True)
    conv_grad = custom_cnn2d(2, 4, 3, 1, 1)
    output = conv_grad(x_grad)
    output.sum().backward()
    print(f"Gradients work: {x_grad.grad is not None}")

    print("✓ Basic tests passed!")


# DEBUGGING QUESTIONS TO ASK YOURSELF:

def debug_questions():
    """
    Questions to ask yourself as you implement:

    INITIALIZATION (__init__):
    1. What shape should self.weight be? Think about the math:
       - You need out_channels filters
       - Each filter processes in_channels input channels
       - Each filter has spatial dimensions kernel_size x kernel_size
       - So weight shape should be: (out_channels, in_channels, kernel_height, kernel_width)

    2. How should you handle kernel_size, stride, padding if they're given as int vs tuple?
       - Hint: If int, convert to (int, int)

    3. What initialization should you use for weights?
       - Look at how Linear layer in basic_blocks.py initializes weights
       - Or research Xavier/He initialization

    FORWARD PASS:
    1. How do you add padding to input?
       - torch.nn.functional.pad might be useful

    2. How can you use unfold to extract sliding windows?
       - unfold extracts patches from input tensor
       - What parameters does it need?

    3. How do you convert convolution to matrix multiplication?
       - After unfolding, you get patches of shape (batch, channels*kernel_h*kernel_w, num_patches)
       - Your weights are (out_channels, in_channels*kernel_h*kernel_w) after reshaping
       - Matrix multiply these together!

    4. How do you reshape the result back to proper spatial dimensions?
       - You need to figure out height_out and width_out from the formula above

    TESTING:
    1. Start simple: implement just the case where kernel_size=1, stride=1, padding=0
    2. Test with small tensors and print intermediate shapes
    3. Compare your output shapes with torch.nn.Conv2d
    4. Once shapes are right, compare actual values
    """
    pass


if __name__ == "__main__":
    print("=== CS336 CNN Implementation Exercise ===")
    print("Read the problem description above, then implement the TODOs!")
    print("Run this file to test your implementation.")
    print("\nTo get started, think about these key questions:")
    print("1. What should the weight tensor shape be?")
    print("2. How can you use torch.nn.functional.unfold?")
    print("3. How can you turn convolution into matrix multiplication?")
    print("\nGood luck! 🎯")
    print("\n" + "="*50)

    test_custom_cnn2d()

