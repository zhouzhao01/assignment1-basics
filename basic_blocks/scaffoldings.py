import torch
import os
from typing import IO, Any, BinaryIO

from dataset import plain_dataset
from metrics import cross_entropy_loss
from basic_blocks import transformer_lm

from optimizer import AdamW
from optimizer import grad_clip


import numpy as np

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import time

from tqdm import tqdm

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

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
    def __init__(self, model, 
                 optimizer:AdamW, lr_schudule:torch.optim.lr_scheduler.CosineAnnealingLR, 
                 dataset:plain_dataset, epoch:int, batch_size:int, 
                 device):
        
        self.model = model
        self.optimizer = optimizer
        self.lr_schudule = lr_schudule
        self.dataset = dataset
        self.epoch = epoch
        self.batch_size = batch_size
        self.device = device

        self.iterations = 0
        self.steps = 0

        # TensorBoard setup
        run_name = f"exp_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        self.writer = SummaryWriter(f"runs/{run_name}")
        self.grad_clip = grad_clip(max_l2_norm=1)

        # For throughput tracking
        self.start_time = None


    def train_batch(self):
        # Timing start
        if self.start_time is None:
            self.start_time = time.time()

        self.optimizer.zero_grad()

        model = self.model.to(self.device)
        input_sequence, targets = self.dataset.get_batch()
        input_sequence = input_sequence.to(self.device)
        targets = targets.to(self.device)
        rlt = model(input_sequence)

        loss_fun = cross_entropy_loss()
        loss = loss_fun(rlt, targets)
        loss.backward()

        self.iterations += self.batch_size # samples
        self.steps += 1 # update steps

        # Gradient norm computation & Gradient Clip.
        total_norm = self.grad_clip.cal_total_l2norm(model.parameters())
        self.grad_clip(model.parameters())

        # self.optimizer.step(self.steps)
        self.optimizer.step()
        self.lr_schudule.step()
        
        # Logging
        self.writer.add_scalar("Loss/train", loss.item(), self.steps)
        # self.writer.add_scalar("Learning_Rate", self.optimizer.lr_schedule(self.steps), self.steps)
        self.writer.add_scalar("Gradient/norm", total_norm, self.steps)

        # Throughput (optional - every N steps?)
        elapsed = time.time() - self.start_time
        tokens_processed = self.iterations * self.model.context_length  # You already track this
        throughput = tokens_processed / elapsed
        self.writer.add_scalar("Throughput/tokens_per_sec", throughput, self.steps)

        return loss

    def valid_batch(self):
        pass

    def test(self):
        pass

def training_together():
    vocab_size:int = 32000
    context_length:int = 128
    num_layers:int = 6
    d_model: int = 512
    num_heads: int = 8
    d_ff: int = 2048
    rope_theta: float = 10000.0

    epoch = 4
    batch_size = 32
    device = "cuda"

    # psedo_data = torch.randint(0,vocab_size,(10000,))
    data_path = "/mnt/aat/zzhao.zhou/cs336_2025/assignment1-basics/data/owt_valid_encodings.npy"
    data = np.memmap(data_path,dtype=np.int32)
    plain_dataset_ins = plain_dataset(data,
                                      batch_size,
                                      context_length,
                                      device)

    tokens_per_batch = batch_size * context_length
    total_tokens = plain_dataset_ins.__len__()
    total_batches = total_tokens // tokens_per_batch * epoch

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
    
    optimizer = torch.optim.AdamW(params=model.parameters(recurse=True))
    lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_batches,
                eta_min=6e-5,
                last_epoch=-1)

    trainer_ins = trainer(model=model, 
                          optimizer=optimizer,
                          lr_schudule=lr_schedule,
                          dataset=plain_dataset_ins, 
                          epoch=epoch, 
                          batch_size=batch_size,
                          device=device)

    
    for batch_index in tqdm(range(total_batches)):
        trainer_ins.train_batch()
        
        if batch_index % 1000 == 0:
                save_checkpoint(
                    model = model,
                    optimizer=optimizer,
                    iteration=batch_index,
                    out=f"/mnt/aat/zzhao.zhou/cs336_2025/assignment1-basics/weights/1016/step_{batch_index}.pt"
                )
    
    trainer_ins.writer.close()
    save_checkpoint(
        model = model,
        optimizer=optimizer,
        iteration=batch_index,
        out=f"/mnt/aat/zzhao.zhou/cs336_2025/assignment1-basics/weights/1016/step_{batch_index}.pt"
    )

def debug_1016():
    vocab_size:int = 32000
    context_length:int = 128
    num_layers:int = 6
    d_model: int = 512
    num_heads: int = 8
    d_ff: int = 2048
    rope_theta: float = 10000.0

    epoch = 1
    batch_size = 32
    device = "cuda"

    # psedo_data = torch.randint(0,vocab_size,(10000,))
    data_path = "/mnt/aat/zzhao.zhou/cs336_2025/assignment1-basics/data/owt_valid_encodings.npy"
    data = np.memmap(data_path,dtype=np.int32)
    plain_dataset_ins = plain_dataset(data,
                                      batch_size,
                                      context_length,
                                      device)

    tokens_per_batch = batch_size * context_length
    total_tokens = plain_dataset_ins.__len__()
    total_batches = total_tokens // tokens_per_batch * epoch

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

    # optimizer = AdamW(model.parameters(recurse=True),flag_lr_schedule=True,
    #                 warmup_iters=int(total_batches*0.05),
    #                 cosine_cycle_iters=total_batches)
    
    optimizer = torch.optim.AdamW(
        params=model.parameters(recurse=True),
    )

    lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                T_max=total_batches,
                eta_min=6e-5,
                last_epoch=-1)

    trainer_ins = trainer(model=model, 
                          optimizer=optimizer,
                          dataset=plain_dataset_ins, 
                          epoch=epoch, 
                          batch_size=batch_size,
                          device=device)
    
    ckp_path = "/mnt/aat/zzhao.zhou/cs336_2025/assignment1-basics/weights/1014/step_41000.pt"
    
    iteration = load_checkpoint(ckp_path,model=model,optimizer=optimizer)
    for batch_index in tqdm(range(iteration, total_batches)):
        loss = trainer_ins.train_batch()
        iteration += 1
        # print(loss)
        if batch_index % 1000 == 0:
                save_checkpoint(
                    model = model,
                    optimizer=optimizer,
                    iteration=total_batches,
                    out=f"/mnt/aat/z zhao.zhou/cs336_2025/assignment1-basics/weights/1016/step_{iteration}.pt"
                )
    
    trainer_ins.writer.close()
    save_checkpoint(
        model = model,
        optimizer=optimizer,
        iteration=total_batches,
        out=f"/mnt/aat/zzhao.zhou/cs336_2025/assignment1-basics/weights/1016/step_{iteration}.pt"
    )

if __name__ == "__main__":
    set_random_seed(seed=42)
    training_together()