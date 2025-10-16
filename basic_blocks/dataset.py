import torch
import numpy.typing as npt

import math
from collections import Counter
import numpy as np
import pytest

class plain_dataset():
    def __init__(self, dataset: npt.NDArray, batch_size: int, context_length: int, device: str):
        self.dataset = dataset
        self.batch_size = batch_size
        self.context_length = context_length
        self.device = device

    def __len__(self):
        return self.dataset.__len__()

    def get_batch(self):
        """
        Returns:
            Tuple of torch.LongTensors of shape (batch_size, context_length). 
            The first tuple item is the sampled input sequences, 
            and the second tuple item is the corresponding language modeling labels.
        """
        valid_sample_num = self.__len__() - self.context_length

        input_seq = torch.zeros(self.batch_size, self.context_length, dtype=torch.long)
        target = torch.zeros(self.batch_size, self.context_length, dtype=torch.long)

        start_list = torch.randint(0, valid_sample_num, (self.batch_size,))
        for sample_index in range(self.batch_size):
            start = start_list[sample_index]
            end   = start + self.context_length            
            input_seq[sample_index,:] = torch.tensor(self.dataset[start:end])
            target[sample_index,:] = torch.tensor(self.dataset[start+1:end+1])

        return input_seq.to(self.device), target.to(self.device)

def test_get_batch():
    dataset = np.arange(0, 100)
    context_length = 7
    batch_size = 32
    device = "cpu"

    # Sanity check to make sure that the random samples are indeed somewhat random.
    starting_indices = Counter()
    num_iters = 1000
    for _ in range(num_iters):
        x, y = run_get_batch(
            dataset=dataset,
            batch_size=batch_size,
            context_length=context_length,
            device=device,
        )

        # Make sure the shape is correct
        assert x.shape == (batch_size, context_length)
        assert y.shape == (batch_size, context_length)

        # Make sure the y's are always offset by 1
        np.testing.assert_allclose((x + 1).detach().numpy(), y.detach().numpy())

        starting_indices.update(x[:, 0].tolist())

    # Make sure we never sample an invalid start index
    num_possible_starting_indices = len(dataset) - context_length
    assert max(starting_indices) == num_possible_starting_indices - 1
    assert min(starting_indices) == 0
    # Expected # of times that we see each starting index
    expected_count = (num_iters * batch_size) / num_possible_starting_indices
    standard_deviation = math.sqrt(
        (num_iters * batch_size) * (1 / num_possible_starting_indices) * (1 - (1 / num_possible_starting_indices))
    )
    # Range for expected outcomes (mu +/- 5sigma). For a given index,
    # this should happen 99.99994% of the time of the time.
    # So, in the case where we have 93 possible start indices,
    # the entire test should pass with 99.9944202% of the time
    occurrences_lower_bound = expected_count - 5 * standard_deviation
    occurrences_upper_bound = expected_count + 5 * standard_deviation

    for starting_index, count in starting_indices.items():
        if count < occurrences_lower_bound:
            raise ValueError(
                f"Starting index {starting_index} occurs {count} times, but expected at least {occurrences_lower_bound}"
            )
        if count > occurrences_upper_bound:
            raise ValueError(
                f"Starting index {starting_index} occurs {count} times, but expected at most {occurrences_upper_bound}"
            )

    with pytest.raises((RuntimeError, AssertionError)) as excinfo:
        # We're assuming that cuda:99 is an invalid device ordinal.
        # Just adding this here to make sure that the device flag is
        # being handled.
        run_get_batch(
            dataset=dataset,
            batch_size=batch_size,
            context_length=context_length,
            device="cuda:99",
        )
        assert "CUDA error" in str(excinfo.value) or "Torch not compiled with CUDA enabled" in str(excinfo.value)

def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    dataset = plain_dataset(dataset,batch_size,context_length,device)
    return dataset.get_batch()

def test_real_data():
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    owt_tokenizer = Tokenizer(BPE()).from_file("/mnt/aat/zzhao.zhou/cs336_2025/assignment1-basics/basic_blocks/tokenizer_owt.json")

    data_path = "/mnt/aat/zzhao.zhou/cs336_2025/assignment1-basics/data/owt_valid_encodings.npy"
    data = np.memmap(data_path,dtype=np.int32)
    plain_dataset_ins = plain_dataset(data,
                                      batch_size=10,
                                      context_length=50,
                                      device='cpu')
    for _ in range(10):
        input_seq, target_seq = plain_dataset_ins.get_batch()
        print(owt_tokenizer.decode(input_seq[0].tolist()).replace("Ġ"," "))

if __name__ == "__main__":
    # test_get_batch()
    test_real_data()
