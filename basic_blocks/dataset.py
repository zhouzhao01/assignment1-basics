import torch
import numpy.typing as npt

class plain_dataset():
    def __init__(self, dataset: npt.NDArray, batch_size: int, context_length: int, device: str):
        self.dataset = dataset
        self.batch_size = batch_size
        self.context_length = context_length
        self.device = device

        self.pointer = 0
        
    def get_batch_sample(self):
        start = torch.arange(0,self.batch_size) + self.pointer
        end   = torch.arange(0,self.batch_size) + self.pointer + self.context_length
        
        input_sequences = torch.zeros([self.batch_size, self.context_length])
        target          = torch.zeros([self.batch_size, self.context_length])

        for sample_index in range(self.batch_size):
            input_sequences[sample_index,:] = torch.tensor(self.dataset[start[sample_index]  :end[sample_index]])
            target[sample_index,:]          = torch.tensor(self.dataset[start[sample_index]+1:end[sample_index]+1])

        self.pointer += self.batch_size * self.context_length
        return input_sequences.to(self.device), target.to(self.device)