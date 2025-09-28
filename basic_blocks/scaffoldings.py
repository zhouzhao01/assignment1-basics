import torch
import os
from typing import IO, Any, BinaryIO

from dataset import plain_dataset
from metrics import cross_entropy_loss
from basic_blocks import transformer_lm
from optimizer import AdamW


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """

    weight_dic = {
        "model_weights":  model.state_dict(),
        "optimizer_weights":  optimizer.state_dict(),
        "iterations": iteration
    }
    torch.save(weight_dic,out)

def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    dic = torch.load(src)
    model.load_state_dict(dic["model_weights"])
    optimizer.load_state_dict(dic["optimizer_weights"])

    return dic["iterations"]

class trainer():
    def __init__(self, model, optimizer:AdamW, dataset:plain_dataset, epoch:int, batch_size:int, device):
        self.model = model
        self.optimizer = optimizer
        self.dataset = dataset
        self.epoch = epoch
        self.batch_size = batch_size
        self.device = device

        self.iterations = 0
        self.steps = 0

    def train_batch(self):
        self.optimizer.zero_grad()

        model = self.model.to(self.device)
        input_sequence, targets = self.dataset.get_batch()
        input_sequence = input_sequence.to(self.device)
        targets = targets.to(self.device)
        rlt = model(input_sequence)

        loss_fun = cross_entropy_loss()
        loss = loss_fun(rlt, targets)
        loss.backward()
        self.optimizer.step(self.steps)

        self.iterations += self.batch_size
        self.steps += 1
        return loss

    def valid_batch(self):
        pass

    def test(self):
        pass

def training_together():
    vocab_size:int = 1024
    context_length:int = 32
    num_layers:int = 4
    d_model: int = 300
    num_heads: int = 25
    d_ff: int = 300
    rope_theta: float = 10000.0

    device = "cuda"

    model = transformer_lm(
        vocab_size,
        context_length,
        num_layers,
        d_model,
        num_heads,
        d_ff,
        rope_theta,
        device
    )
    optimizer = AdamW(model.parameters(recurse=True))

    epoch = 4
    batch_num = 4
    batch_size = 32
    
    psedo_data = torch.randint(0,vocab_size,(10000,))

    plain_dataset_ins = plain_dataset(psedo_data,
                                      batch_size,
                                      context_length,
                                      device)

    trainer_ins = trainer(model=model, 
                          optimizer=optimizer,
                          dataset=plain_dataset_ins, 
                          epoch=epoch, 
                          batch_size=batch_size,
                          device=device)

    for epoch_index in range(epoch):
        for batch_index in range(batch_num):
            loss = trainer_ins.train_batch()
            print(loss)

if __name__ == "__main__":
    training_together()
