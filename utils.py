# utils.py (Checkpoint 관리 및 로드)
import os
import torch
import config


def save_checkpoint(state: dict, is_best: bool = False, filename: str = "checkpoint.pt"):
    os.makedirs(config.SAVE_DIR, exist_ok=True)
    filepath = os.path.join(config.SAVE_DIR, filename)
    torch.save(state, filepath)
    if is_best:
        best_path = os.path.join(config.SAVE_DIR, "best_model.pt")
        torch.save(state, best_path)


def load_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer = None, filename: str = None):
    filepath = filename or os.path.join(config.SAVE_DIR, "checkpoint.pt")
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint.get("state_dict", checkpoint))
    if optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint