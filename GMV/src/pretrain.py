from src.data import get_pretrain_dataset
from src.dataset import TrainingDataset, EvaluationDataset
from torch.utils.data import DataLoader
from src.models.aion import AION
from src.trainer import PretrainedTrainer
from logging import getLogger
import torch

logger = getLogger()


def log_model_size(logger, model: torch.nn.Module, device=None, prefix=""):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    total_bytes = param_bytes + buffer_bytes

    logger.info(
        f"{prefix}Model size: "
        f"params={total_params:,} ({total_params/1e6:.3f}M), "
        f"trainable={trainable_params:,} ({trainable_params/1e6:.3f}M), "
        f"param_mem={param_bytes/1024**2:.2f} MB, "
        f"total_mem={total_bytes/1024**2:.2f} MB"
    )

    if device is not None and torch.cuda.is_available() and "cuda" in str(device):
        torch.cuda.synchronize(device)
        alloc = torch.cuda.memory_allocated(device) / 1024**2
        reserved = torch.cuda.memory_reserved(device) / 1024**2
        logger.info(f"{prefix}CUDA snapshot: allocated={alloc:.1f} MB, reserved={reserved:.1f} MB")


def run(args, device):
    """Run AION pretraining."""
    dataset = get_pretrain_dataset(args)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    train_dataset = TrainingDataset(train_dataset)
    test_dataset = EvaluationDataset(test_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    groups = 3
    model = AION(args, groups=groups).to(device)
    logger.info(f"Model:\n {model}")
    log_model_size(logger, model, device=device, prefix=f"[{args.model}] ")

    trainer = PretrainedTrainer(args, model, device)
    logger.info("Start training...")
    trainer.train(train_dataloader, test_dataloader)
