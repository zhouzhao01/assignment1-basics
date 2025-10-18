import torch
import os
from typing import IO, Any, BinaryIO
import json
from pathlib import Path

from dataset import plain_dataset
from dataset import word_dataset
from torch.utils.data import DataLoader

from metrics import cross_entropy_loss
from basic_blocks import transformer_lm

from optimizer import AdamW
from optimizer import grad_clip


import numpy as np

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import time
import wandb

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
        current_time = time.strftime("%Y-%m-%d_%H-%M-%S")
        return base_dir / self.config['exp_name'] / f"exp_{current_time}"

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
            device=self.config['device'],
            use_fast_attn=m['use_fast_attn'],
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
    
    def build_torch_dataset(self) -> word_dataset:
        data_path = self.config['data']['train_dataset']
        data = np.memmap(data_path, dtype=np.int32, mode='r')

        return word_dataset(
            dataset=data,
            # batch_size=self.config['training']['batch_size'],
            context_length=self.config['model']['context_length'],
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

    def build_trainer(self, model, optimizer, lr_schedule, dataloader, exp_dir: Path):
        """Build trainer instance from config"""
        return trainer(
            model=model,
            optimizer=optimizer,
            lr_schedule=lr_schedule,
            dataloader=dataloader,
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
    model.load_state_dict(dic["model_weights"], strict=False)
    optimizer.load_state_dict(dic["optimizer_weights"])

    return dic["iterations"]

class trainer():
    def __init__(self, model,
                 optimizer:AdamW, lr_schedule:torch.optim.lr_scheduler.CosineAnnealingLR,
                 dataloader:DataLoader, epoch:int, batch_size:int,
                 device, exp_name:str, exp_dir:str, config:dict):

        self.model = model
        self.optimizer = optimizer
        self.lr_schedule = lr_schedule
        self.dataloader = dataloader
        self.epoch = epoch
        self.batch_size = batch_size
        self.device = device
        self.exp_name = exp_name
        self.exp_dir = Path(exp_dir)
        self.config = config
        self.loss_fun = cross_entropy_loss()

        self.iterations = 0
        self.steps = 0

        # TensorBoard setup - use exp_dir for unified naming
        self.writer = SummaryWriter(str(self.exp_dir))
        self.wandb_logger = wandb.init(
            project= "assignment1",
            config=config
        )

        self.grad_clip = grad_clip(max_l2_norm=1)

        # For throughput tracking
        self.start_time = None

        # Profiling setup
        self.profiling_enabled = config.get('profiling', {}).get('enabled', False)
        self.profiling_log_interval = config.get('profiling', {}).get('log_every_n_steps', 10)
        if self.profiling_enabled:
            self.profile_timers = {
                'data_transfer': 0.0,
                'forward': 0.0,
                'loss_computation': 0.0,
                'backward': 0.0,
                'grad_clip': 0.0,
                'optimizer_step': 0.0,
                'logging': 0.0
            }


    def train_batch(self, input_sequence, targets):
        # Timing start
        if self.start_time is None:
            self.start_time = time.time()

        # === PROFILING: Data Transfer ===
        if self.profiling_enabled:
            torch.cuda.synchronize()
            t0 = time.time()

        self.optimizer.zero_grad()
        input_sequence = input_sequence.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)

        if self.profiling_enabled:
            torch.cuda.synchronize()
            self.profile_timers['data_transfer'] += time.time() - t0

        # === PROFILING: Forward Pass ===
        if self.profiling_enabled:
            torch.cuda.synchronize()
            t0 = time.time()

        rlt = self.model(input_sequence)

        if self.profiling_enabled:
            torch.cuda.synchronize()
            self.profile_timers['forward'] += time.time() - t0

        # === PROFILING: Loss Computation ===
        if self.profiling_enabled:
            torch.cuda.synchronize()
            t0 = time.time()

        
        loss = self.loss_fun(rlt, targets)

        if self.profiling_enabled:
            torch.cuda.synchronize()
            self.profile_timers['loss_computation'] += time.time() - t0

        # === PROFILING: Backward Pass ===
        if self.profiling_enabled:
            torch.cuda.synchronize()
            t0 = time.time()

        loss.backward()

        if self.profiling_enabled:
            torch.cuda.synchronize()
            self.profile_timers['backward'] += time.time() - t0

        self.iterations += self.batch_size # samples
        self.steps += 1 # update steps

        # === PROFILING: Gradient Clipping ===
        if self.profiling_enabled:
            torch.cuda.synchronize()
            t0 = time.time()

        total_norm = self.grad_clip.cal_total_l2norm(self.model.parameters())
        self.grad_clip(self.model.parameters())

        if self.profiling_enabled:
            torch.cuda.synchronize()
            self.profile_timers['grad_clip'] += time.time() - t0

        # === PROFILING: Optimizer Step ===
        if self.profiling_enabled:
            torch.cuda.synchronize()
            t0 = time.time()

        self.optimizer.step()
        self.lr_schedule.step()

        if self.profiling_enabled:
            torch.cuda.synchronize()
            self.profile_timers['optimizer_step'] += time.time() - t0

        # Throughput
        elapsed = time.time() - self.start_time
        tokens_processed = self.iterations * self.model.context_length
        throughput = tokens_processed / elapsed

        # === PROFILING: Logging ===
        if self.profiling_enabled:
            t0 = time.time()

        self.writer.add_scalar("Loss/train", loss.item(), self.steps)
        self.writer.add_scalar("Gradient/norm", total_norm, self.steps)
        self.writer.add_scalar("Throughput/tokens_per_sec", throughput, self.steps)

        self.wandb_logger.log(
            {
                "Loss/train": loss.item(), 
                "Gradient/norm": total_norm,
                "Throughput/tokens_per_sec": throughput
            }
        )
        

        if self.profiling_enabled:
            self.profile_timers['logging'] += time.time() - t0

            # Print profiling summary every N steps
            if self.steps % self.profiling_log_interval == 0:
                self._print_profile_summary()

        return loss

    def _print_profile_summary(self):
        """Print profiling summary and reset timers"""
        total_time = sum(self.profile_timers.values())
        print(f"\n{'='*60}")
        print(f"Profiling Summary (Steps {self.steps - self.profiling_log_interval + 1}-{self.steps})")
        print(f"{'='*60}")
        print(f"{'Stage':<25} {'Time (s)':<12} {'Percentage':<12}")
        print(f"{'-'*60}")

        for stage, time_spent in self.profile_timers.items():
            percentage = (time_spent / total_time * 100) if total_time > 0 else 0
            print(f"{stage:<25} {time_spent:<12.4f} {percentage:<12.2f}%")

        print(f"{'-'*60}")
        print(f"{'Total':<25} {total_time:<12.4f} {'100.00%':<12}")
        print(f"{'='*60}\n")

        # Reset timers for next interval
        for key in self.profile_timers:
            self.profile_timers[key] = 0.0

    def valid_batch(self):
        pass

    def test(self):
        pass

def training_together(config_path: str):
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
    # Ensure model is on GPU and in training mode
    model = model.to(builder_ins.config['device'])
    model.train()

    dataset = builder_ins.build_torch_dataset()

    train_loader = DataLoader(
            dataset,
            batch_size=builder_ins.config["training"]["batch_size"],
            shuffle=True,
            num_workers=4,  # KEY: parallel data loading
            pin_memory=True,  # faster CPU->GPU transfer
            persistent_workers=True  # keep workers alive between epochs
        )


    optimizer = builder_ins.build_optimizer(model)
    total_batches = builder_ins.calculate_total_batches()
    lr_schedule = builder_ins.build_lr_schedule(optimizer, total_batches)
    trainer_ins = builder_ins.build_trainer(model, optimizer, lr_schedule, train_loader, exp_dir)

    print(f"Total batches: {total_batches}")
    print(f"Starting training for {builder_ins.config['training']['epoch']} epochs...")

    # Training loop
    save_every_n = builder_ins.config['checkpoint']['save_every_n_steps']
    profiling_enabled = builder_ins.config.get('profiling', {}).get('enabled', False)
    profiling_log_interval = builder_ins.config.get('profiling', {}).get('log_every_n_steps', 10)

    if profiling_enabled:
        dataloader_time = 0.0
        train_time = 0.0
        dataloader_start = None
    
    trainer_ins.wandb_logger.watch(model, trainer_ins.loss_fun)

    for epoch in range(builder_ins.config['training']['epoch']):
        dataloader_iter = iter(train_loader)
        for batch_idx in tqdm(range(total_batches), desc=f"Epoch {epoch+1}"):

            if trainer_ins.steps >= total_batches:
                break

            # Measure dataloader time
            if profiling_enabled:
                dataloader_fetch_start = time.time()

            input_sequence, targets = next(dataloader_iter)

            if profiling_enabled:
                dataloader_time += time.time() - dataloader_fetch_start

            # Measure training time
            if profiling_enabled:
                train_start = time.time()

            trainer_ins.train_batch(input_sequence, targets)

            if profiling_enabled:
                train_time += time.time() - train_start

                # Print dataloader vs training comparison every N steps
                if trainer_ins.steps % profiling_log_interval == 0 and trainer_ins.steps > 0:
                    total = dataloader_time + train_time
                    print(f"\n{'='*60}")
                    print(f"DataLoader vs Training Time (Steps {trainer_ins.steps - profiling_log_interval + 1}-{trainer_ins.steps})")
                    print(f"{'='*60}")
                    print(f"DataLoader waiting:  {dataloader_time:.4f}s ({dataloader_time/total*100:.2f}%)")
                    print(f"Training compute:    {train_time:.4f}s ({train_time/total*100:.2f}%)")
                    print(f"{'='*60}\n")
                    dataloader_time = 0.0
                    train_time = 0.0

            if trainer_ins.steps % save_every_n == 0 and trainer_ins.steps > 0:
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    iteration=trainer_ins.steps,
                    out=exp_dir / f"step_{trainer_ins.steps}.pt"
                )

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
    training_together("configs/config_4090.json")