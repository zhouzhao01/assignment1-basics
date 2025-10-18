import torch
import os
import subprocess

from typing import IO, Any, BinaryIO
import json
from pathlib import Path

from dataset import plain_dataset
from dataset import TokenDataset
from torch.utils.data import DataLoader

from metrics import cross_entropy_loss

from optimizer import AdamW, grad_clip, lr_cosine_schedule

from models import transformer_lm

import numpy as np

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import time
import wandb

from tqdm import tqdm

import yaml

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
        if config_path is None:
            config_path = "/mnt/aat/zzhao.zhou/cs336_2025/assignment1-basics/configs/config_4090.json"
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
        batches_per_epoch = dataset_size // tokens_per_batch * epoch
        return batches_per_epoch 

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
    
    def build_torch_dataset(self, data_split: str) -> TokenDataset:
        """
        Build a torch dataset.
            data_split: str. Choose from ["train", "valid"].
        """
        data_path = self.config['data'][f'{data_split}_dataset']
        # data = np.memmap(data_path, dtype=np.int32, mode='r')
        data = np.load(file=data_path,mmap_mode='r')
        return TokenDataset(
            tokens=data,
            context_length=self.config['model']['context_length'],
        )

    def build_optimizer(self, model: torch.nn.Module) -> torch.optim.AdamW:
        """Build AdamW optimizer from config"""
        opt = self.config['optimizer']
        if opt["type"] == "custom_AdamW":
            optimizer = AdamW(
                params=model.parameters(),
                lr=opt['lr'],
                betas=tuple(opt['betas']),
                weight_decay=opt['weight_decay'],
                eps=opt['eps']
                )
        else:
            optimizer = torch.optim.AdamW(
                params=model.parameters(),
                lr=opt['lr'],
                betas=tuple(opt['betas']),
                eps=opt['eps'],
                weight_decay=opt['weight_decay']
            )
        return optimizer
    
    def build_lr_schedule(self, optimizer: torch.optim.Optimizer, total_batches: int) -> torch.optim.lr_scheduler.CosineAnnealingLR:
        """Build learning rate scheduler from config"""
        sched = self.config['lr_scheduler']
        if sched["type"] == "custom_CosineAnnealingLR":
            total_batches = self.calculate_total_batches() * self.config["training"]["epoch"]
            warmup_iters= sched["warmup_iters"]
            cosine_cycle_iters = total_batches - warmup_iters
            lr_scheduler = lr_cosine_schedule(
                optimizer=optimizer,
                max_learning_rate=sched["max_learning_rate"],
                min_learning_rate=sched["min_learning_rate"],
                warmup_iters= warmup_iters,
                cosine_cycle_iters= cosine_cycle_iters
            )
        else:
            lr_scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                T_max=total_batches,
                eta_min=sched['min_learning_rate'],
                last_epoch=-1
            )
        return lr_scheduler

    def build_trainer(self, model, optimizer, lr_schedule, train_dataloader, valid_dataloader, exp_dir: Path, 
                      wandb_runs:wandb.Run):
        """Build trainer instance from config"""
        return trainer(
            model=model,
            optimizer=optimizer,
            lr_schedule=lr_schedule,
            train_dataloader=train_dataloader,
            valid_dataloader=valid_dataloader,
            epoch=self.config['training']['epoch'],
            batch_size=self.config['training']['batch_size'],
            device=self.config['device'],
            exp_name=self.config['exp_name'],
            exp_dir=str(exp_dir),
            config=self.config,
            wandb_runs=wandb_runs
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
    def __init__(self, model:transformer_lm,
                optimizer:AdamW, lr_schedule:torch.optim.lr_scheduler.CosineAnnealingLR,
                train_dataloader:DataLoader, epoch:int, batch_size:int,
                valid_dataloader:DataLoader,
                device, exp_name:str, exp_dir:str, config:dict,
                wandb_runs: wandb.Run
                ):

        self.model = model
        self.optimizer = optimizer
        self.lr_schedule = lr_schedule

        self.train_dataloader = train_dataloader
        self.epoch = epoch
        self.batch_size = batch_size

        self.valid_dataloader = valid_dataloader

        self.device = device
        self.exp_name = exp_name
        self.exp_dir = Path(exp_dir)
        self.config = config
        self.loss_fun = cross_entropy_loss()

        self.iterations = 0
        self.steps = 0

        # TensorBoard setup - use exp_dir for unified naming
        # self.writer = SummaryWriter(str(self.exp_dir))
        self.wandb_logger = wandb_runs

        self.grad_clip = grad_clip(max_l2_norm=1)

        # For throughput tracking
        self.start_time = None

    def train_batch(self, input_sequence:torch.Tensor, targets:torch.Tensor):
        if self.start_time is None:
            self.start_time = time.time()
        
        self.model.train()
        self.optimizer.zero_grad()

        input_sequence = input_sequence.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)

        rlt = self.model(input_sequence)

        loss = self.loss_fun(rlt, targets)

        loss.backward()

        self.iterations += self.batch_size # samples
        self.steps += 1 # update steps

        total_norm = self.grad_clip.cal_total_l2norm(self.model.parameters())
        self.grad_clip(self.model.parameters())

        self.optimizer.step()
        self.lr_schedule.step()

        # Throughput
        elapsed = time.time() - self.start_time
        tokens_processed = self.iterations * self.model.context_length
        throughput = tokens_processed / elapsed

        self.wandb_logger.log(
            {
                "Loss/train": loss.item(),
                "Gradient/norm": total_norm,
                "Throughput/tokens_per_sec": throughput
            }
        )

        return loss
    
    def valid(self) -> torch.Tensor:
        """
        Valid on a whole valid dataset
        """
       
        self.model.eval()
        total_valid_batch = self.valid_dataloader.__len__()
        
        with torch.no_grad():
            loss = 0
            for valid_batch_index, (input_sequence, targets) in enumerate(self.valid_dataloader):
                # print(f"valid_batch_index: {valid_batch_index}")
                loss += self.valid_batch(input_sequence, targets)
            
            loss = loss / total_valid_batch

        self.wandb_logger.log(
            {
                "Loss/valid": loss.item(),
            }
        )
        
        return loss


    def valid_batch(self, input_sequence:torch.Tensor, targets:torch.Tensor):
        input_sequence = input_sequence.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)
        rlt = self.model(input_sequence)
        loss = self.loss_fun(rlt, targets)
        return loss


def training_together(config_path:str=None):
    """
    Config-driven training function.

    Args:
        config_path: Path to config JSON file
    """

    with open("/mnt/aat/zzhao.zhou/cs336_2025/assignment1-basics/sweep_config.yaml") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
        
    wandb_runs = wandb.init(config=config)

    # Load config and setup
    builder_ins = builder(config_path)

    # wandb sweep hyperparameter fetch
    builder_ins.config["training"]["batch_size"] = wandb_runs.config["batch_size"]

    builder_ins.config["lr_scheduler"]["max_learning_rate"] = wandb_runs.config["max_learning_rate"]
    builder_ins.config["lr_scheduler"]["min_learning_rate"] = wandb_runs.config["max_learning_rate"] * wandb_runs.config["min_lr_ratio"]
    builder_ins.config["lr_scheduler"]["warmup_iters"] = wandb_runs.config["warmup_iters"]
    
    builder_ins.config["optimizer"]["weight_decay"] = wandb_runs.config["weight_decay"]

    # print(f"builder_ins.config: {builder_ins.config}")
    # builder_ins.config["optimizer"]["betas"] = wandb_runs.config["weight_decay"]

    set_random_seed(builder_ins.config['seed'])
    # Setup experiment directory
    exp_dir = builder_ins.setup_experiment_dir()
    print(f"Experiment directory: {exp_dir}")

    # Build all components
    model = builder_ins.build_model()
    train_dataset = builder_ins.build_torch_dataset(data_split="train")
    train_loader = DataLoader(
            train_dataset,
            batch_size=builder_ins.config["training"]["batch_size"],
            shuffle=True,
            num_workers=4,  # KEY: parallel data loading
            pin_memory=True,  # faster CPU->GPU transfer
            persistent_workers=True  # keep workers alive between epochs
    )
    valid_dataset = builder_ins.build_torch_dataset(data_split="valid")
    valid_loader = DataLoader(
            valid_dataset,
            batch_size=builder_ins.config["training"]["batch_size"],
            shuffle=True,
            num_workers=4,  # KEY: parallel data loading
            pin_memory=True,  # faster CPU->GPU transfer
            persistent_workers=True  # keep workers alive between epochs
    )
    optimizer = builder_ins.build_optimizer(model)
    total_batches = builder_ins.calculate_total_batches()
    lr_schedule = builder_ins.build_lr_schedule(optimizer, total_batches)
    trainer_ins = builder_ins.build_trainer(model, optimizer, lr_schedule, train_loader, valid_loader, exp_dir, wandb_runs)

    print(f"Total training batches: {total_batches}")
    print(f"Starting training for {builder_ins.config['training']['epoch']} epochs...")

    # Training Loop
    best_loss = None
    save_every_n = builder_ins.config['checkpoint']['save_every_n_steps']
    trainer_ins.wandb_logger.watch(model, trainer_ins.loss_fun)
    model = model.to(builder_ins.config['device'])
    for epoch in range(builder_ins.config['training']['epoch']):
        train_dataloader_iter = iter(train_loader)
        
        for _ in tqdm(range(total_batches), desc=f"Epoch {epoch+1}"):
            # Train Model
            model.train()
            input_sequence, targets = next(train_dataloader_iter)
            trainer_ins.train_batch(input_sequence, targets)

            # Valid Model
            # trainer_ins.valid()
            if trainer_ins.steps % save_every_n == 0 and trainer_ins.steps > 0:
                loss = trainer_ins.valid()

                if best_loss == None:
                    best_loss = loss

                elif loss < best_loss:
                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        iteration=trainer_ins.steps,
                        out=exp_dir / f"best.pt"
                    )
                    best_loss = loss

    # Final checkpoint and cleanup
    # trainer_ins.writer.close()
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        iteration=total_batches,
        out=exp_dir / f"step_{total_batches}.pt"
    )
    print(f"Training complete. Final checkpoint saved to {exp_dir / f'step_{total_batches}.pt'}")
    trainer_ins.wandb_logger.finish()

if __name__ == "__main__":
    # sweep_configuration = {
    #     "method": "random",
    #     "name": "assignment1-optimizer-lr",
    #     "description": "sweep optimizer and batchsize hyperparameters",
    #     "metric": {
    #         "goal": "minimize", 
    #         "name": "Loss/train"
    #     },
    #     "parameters": {
    #         "batch_size": {
    #             "values": [1, 16, 32, 64, 128, 256]
    #         },

    #         "max_learning_rate": {
    #             "values": [1e-4, 3e-4, 5e-4, 1e-3]
    #         },
    #         "min_lr_ratio": {
    #             "values": [0.01, 0.05, 0.1]
    #         },
    #         "warmup_iters": {
    #             "values": [500, 1000, 2000, 3000, 4000, 5000]
    #         },
    #         "weight_decay": {
    #             "values": [0.0, 0.01, 0.1]
    #         },
    #     }
    # }

    # # Create sweep
    # sweep_id = wandb.sweep(
    #     sweep_configuration, 
    #     project="assignment1",
    #     entity="push-seminar-4l-hong-kong-university-of-science-and-tech"  # or your entity name
    # )

    training_together("configs/config_4090.json")