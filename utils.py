# utils.py (Checkpoint 관리 및 로드)
import os
import torch
import logging

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


def setup_logging(log_dir="logs", log_name="train.log"):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_name)

    # 기본 로거 설정
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 콘솔 핸들러
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s",
                                     datefmt="%Y-%m-%d %H:%M:%S")
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    # 파일 핸들러
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(ch_formatter)
    logger.addHandler(fh)

    return logger