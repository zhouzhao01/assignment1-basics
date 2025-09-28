import torch
import numpy.typing as npt

class plain_dataset():
    def __init__(self, dataset: npt.NDArray, batch_size: int, context_length: int, device: str):
        self.dataset = dataset
        self.batch_size = batch_size
        self.context_length = context_length
        self.device = device

        self.pointer = 0

    def get_batch(self):
        """
        Returns:
            Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
            is the sampled input sequences, and the second tuple item is the corresponding
            language modeling labels.
        """

        input_seq = torch.zeros(self.batch_size, self.context_length)
        target = torch.zeros(self.batch_size, self.context_length)

        start_list = torch.arange(0, self.batch_size*self.context_length, self.batch_size) + self.pointer

        for sample_index in range(self.batch_size):
            start = start_list[sample_index]
            end = start + self.context_length
            input_seq[sample_index,:] = self.dataset[start:end]
            target[sample_index,:] = self.dataset[start+1:end+1]
        
        self.pointer = self.pointer + self.batch_size * self.context_length
        return input_seq.to(self.device), target.to(self.device)

        