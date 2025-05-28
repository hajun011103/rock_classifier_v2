# train.py
import os
import random
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import f1_score
import datetime
from torch.utils.tensorboard import SummaryWriter

import config
from dataset import stratified_kfold_dataloaders, make_test_dataloader
from model import build_model
from utils import save_checkpoint, load_checkpoint, setup_logging

# SEED
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # 재현성이 중요
    torch.backends.cudnn.benchmark = False     # 속도가 중요

# train loop
def train_one_fold(fold_id, train_loader, val_loader, logger):
    # TensorBoard init
    log_dir = os.path.join("runs", config.RUN_NAME, f"fold{fold_id+1}", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(log_dir=log_dir)

    model = build_model().to(config.DEVICE)
    optimizer = AdamW(model.parameters(),
                      lr=config.LR,
                      weight_decay=config.WEIGHT_DECAY)
    # Cosine Annealing LR scheduler
    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=config.WARM_UP_EPOCHS),
            CosineAnnealingLR(optimizer, T_max=config.EPOCHS - config.WARM_UP_EPOCHS, eta_min=config.MIN_LR)
        ],
        milestones=[config.WARM_UP_EPOCHS]
    )

    criterion = nn.CrossEntropyLoss(weight=config.WEIGHTS)
    scaler = torch.amp.GradScaler(device_type="cuda")

    # Resume checkpoint if exists
    resume_path = os.path.join(config.SAVE_DIR,
                               f"fold{fold_id+1}_best_model.pt")
    if os.path.exists(resume_path):
        logger.info(f"Loading checkpoint for fold {fold_id+1} from {resume_path}")
        load_checkpoint(model, optimizer, filename=resume_path)

    best_val_f1 = -1.0
    patience_counter = 0

    for epoch in range(config.EPOCHS):
        # ----- Train -----
        model.train()
        tr_loss = tr_correct = tr_total = 0
        for imgs, targets in tqdm(train_loader,
                                  desc=f"Fold{fold_id+1} Train E{epoch+1}"):
            imgs, labels = imgs.to(config.DEVICE), targets.to(config.DEVICE)

            optimizer.zero_grad()
            with autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            tr_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            tr_correct += (preds == labels).sum().item()
            tr_total   += labels.size(0)

        train_loss = tr_loss / tr_total
        train_acc  = tr_correct / tr_total

        # ----- Validate -----
        model.eval()
        val_loss = val_correct = val_total = 0
        y_true, y_pred = [], []
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, labels = imgs.to(config.DEVICE), targets.to(config.DEVICE)
                outputs = model(imgs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * imgs.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        val_loss = val_loss / val_total
        val_acc  = val_correct / val_total
        val_f1   = f1_score(y_true, y_pred, average='macro')

        # ----- Scheduler Step -----
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # ----- Logging -----
        logger.info(
            f"[Fold{fold_id+1} Epoch{epoch+1}/{config.EPOCHS}] "
            f"LR: {current_lr:.6f} | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.3f} | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.3f}, F1: {val_f1:.4f}"
        )

        # TensorBoard 기록
        writer.add_scalar('Learning Rate', current_lr, epoch+1)
        writer.add_scalar('Train/Loss', train_loss, epoch+1)
        writer.add_scalar('Train/Accuracy', train_acc, epoch+1)
        writer.add_scalar('Val/Loss', val_loss, epoch+1)
        writer.add_scalar('Val/Accuracy', val_acc, epoch+1)
        writer.add_scalar('Val/Macro_F1', val_f1, epoch+1)

        # ----- Checkpoint & Early Stopping -----
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            save_checkpoint({
                "epoch":      epoch,
                "state_dict": model.state_dict(),
                "optimizer":  optimizer.state_dict()
            }, is_best=True,
               filename=f"fold{fold_id+1}_best_model.pt")
        else:
            patience_counter += 1

        if patience_counter >= config.PATIENCE:
            logger.info("Early stopping triggered.")
            break

    writer.close()


def main():
    # Seed & Logger
    set_seed(config.SEED)
    logger = setup_logging(log_dir="logs", log_name="train.log")

    folds = stratified_kfold_dataloaders()
    for i, (tr, vl) in enumerate(folds):
        train_one_fold(i, tr, vl, logger)

if __name__ == "__main__":
    main()
