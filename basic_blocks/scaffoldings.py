import torch
import os
from typing import IO, Any, BinaryIO
import json
from pathlib import Path

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

class builder():
    """
    Config-driven builder for constructing model, optimizer, dataset, and trainer.
    Stateless design: each build method returns a new instance without storing state.
    """
    def __init__(self, config_path: str):
        """Read from config path, load and validate required fields exist"""
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self._validate_required_fields()

    def _load_config(self, path: str) -> dict:
        """Load JSON config file"""
        with open(path, 'r') as f:
            return json.load(f)

    def _validate_required_fields(self):
        """Check that all required config sections and fields exist"""
        required = ['exp_name', 'seed', 'device', 'model', 'training',
                    'optimizer', 'lr_scheduler', 'data', 'checkpoint']
        for field in required:
            if field not in self.config:
                raise ValueError(f"Missing required config field: {field}")

    def get_exp_dir(self) -> Path:
        """Get experiment directory path (doesn't create it)"""
        base_dir = Path(self.config['checkpoint']['save_dir'])
        return base_dir / self.config['exp_name']

    def setup_experiment_dir(self) -> Path:
        """Create experiment directory and save config.json to it"""
        exp_dir = self.get_exp_dir()
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Save config to experiment directory for reproducibility
        config_save_path = exp_dir / 'config.json'
        with open(config_save_path, 'w') as f:
            json.dump(self.config, f, indent=4)

        return exp_dir

    def get_dataset_size(self) -> int:
        """Load dataset file and return total number of tokens"""
        data_path = self.config['data']['train_dataset']
        data = np.memmap(data_path, dtype=np.int32, mode='r')
        return len(data)

    def calculate_total_batches(self) -> int:
        """Calculate total training batches from config + dataset size"""
        dataset_size = self.get_dataset_size()
        batch_size = self.config['training']['batch_size']
        context_length = self.config['model']['context_length']
        epoch = self.config['training']['epoch']

        tokens_per_batch = batch_size * context_length
        batches_per_epoch = dataset_size // tokens_per_batch
        return batches_per_epoch * epoch

    def build_model(self) -> transformer_lm:
        """Build transformer model from config"""
        m = self.config['model']
        return transformer_lm(
            vocab_size=m['vocab_size'],
            context_length=m['context_length'],
            num_layers=m['num_layers'],
            d_model=m['d_model'],
            num_heads=m['num_heads'],
            d_ff=m['d_ff'],
            rope_theta=m['rope_theta'],
            device=self.config['device']
        )

    def build_dataset(self) -> plain_dataset:
        """Build dataset from config"""
        data_path = self.config['data']['train_dataset']
        data = np.memmap(data_path, dtype=np.int32, mode='r')

        return plain_dataset(
            dataset=data,
            batch_size=self.config['training']['batch_size'],
            context_length=self.config['model']['context_length'],
            device=self.config['device']
        )

    def build_optimizer(self, model: torch.nn.Module) -> torch.optim.AdamW:
        """Build AdamW optimizer from config"""
        opt = self.config['optimizer']
        return torch.optim.AdamW(
            params=model.parameters(),
            lr=opt['lr'],
            betas=tuple(opt['betas']),
            eps=opt['eps'],
            weight_decay=opt['weight_decay']
        )

    def build_lr_schedule(self, optimizer: torch.optim.Optimizer, total_batches: int) -> torch.optim.lr_scheduler.CosineAnnealingLR:
        """Build learning rate scheduler from config"""
        sched = self.config['lr_scheduler']
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=total_batches,
            eta_min=sched['eta_min'],
            last_epoch=-1
        )

    def build_trainer(self, model, optimizer, lr_schedule, dataset, exp_dir: Path):
        """Build trainer instance from config"""
        return trainer(
            model=model,
            optimizer=optimizer,
            lr_schudule=lr_schedule,
            dataset=dataset,
            epoch=self.config['training']['epoch'],
            batch_size=self.config['training']['batch_size'],
            device=self.config['device'],
            exp_name=self.config['exp_name'],
            exp_dir=str(exp_dir),
            config=self.config
        )

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
                 device, exp_name:str, exp_dir:str, config:dict):

        self.model = model
        self.optimizer = optimizer
        self.lr_schudule = lr_schudule
        self.dataset = dataset
        self.epoch = epoch
        self.batch_size = batch_size
        self.device = device
        self.exp_name = exp_name
        self.exp_dir = Path(exp_dir)
        self.config = config

        self.iterations = 0
        self.steps = 0

        # TensorBoard setup - use exp_dir for unified naming
        self.writer = SummaryWriter(str(self.exp_dir))
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

def training_together(config_path: str = "configs/config.json"):
    """
    Config-driven training function.

    Args:
        config_path: Path to config JSON file
    """
    # Load config and setup
    builder_ins = builder(config_path)
    set_random_seed(builder_ins.config['seed'])

    # Setup experiment directory
    exp_dir = builder_ins.setup_experiment_dir()
    print(f"Experiment directory: {exp_dir}")

    # Build all components
    model = builder_ins.build_model()
    dataset = builder_ins.build_dataset()
    optimizer = builder_ins.build_optimizer(model)
    total_batches = builder_ins.calculate_total_batches()
    lr_schedule = builder_ins.build_lr_schedule(optimizer, total_batches)
    trainer_ins = builder_ins.build_trainer(model, optimizer, lr_schedule, dataset, exp_dir)

    print(f"Total batches: {total_batches}")
    print(f"Starting training for {builder_ins.config['training']['epoch']} epochs...")

    # Training loop
    save_every_n = builder_ins.config['checkpoint']['save_every_n_steps']
    for batch_index in tqdm(range(total_batches)):
        trainer_ins.train_batch()

        if batch_index % save_every_n == 0 and batch_index > 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                iteration=batch_index,
                out=exp_dir / f"step_{batch_index}.pt"
            )

            if batch_index == 100:
                break

    # Final checkpoint and cleanup
    trainer_ins.writer.close()
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        iteration=total_batches,
        out=exp_dir / f"step_{total_batches}.pt"
    )
    print(f"Training complete. Final checkpoint saved to {exp_dir / f'step_{total_batches}.pt'}")

if __name__ == "__main__":
    training_together()