import wandb
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import f1_score

import config
from dataset import make_kfold_dataloaders, make_test_dataloader
from model import build_model
from utils import save_checkpoint, load_checkpoint

# AMP context manager alias
from torch.cuda.amp import autocast, GradScaler

def train_one_fold(fold_id, train_loader, val_loader):
    wandb.init(project=config.PROJECT_NAME,
               name=f"{config.RUN_NAME}_fold{fold_id+1}")

    model = build_model().to(config.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()

    best_val_f1 = -1.0

    for epoch in range(config.EPOCHS):
        # ---------- Train ----------
        model.train()
        tr_loss = tr_correct = tr_total = 0
        for imgs, targets in tqdm(train_loader, desc=f"Fold{fold_id+1} Train Epoch{epoch+1}"):
            imgs, targets = imgs.to(config.DEVICE), targets.to(config.DEVICE)
            labels = targets.argmax(dim=1)

            optimizer.zero_grad()
            with autocast():
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

        # ---------- Validate ----------
        model.eval()
        val_loss = val_correct = val_total = 0
        y_true, y_pred = [], []
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(config.DEVICE), targets.to(config.DEVICE)
                labels = targets.argmax(dim=1)
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

        # ---------- Log to WandB ----------
        wandb.log({
            "epoch":         epoch + 1,
            "train_loss":    train_loss,
            "train_acc":     train_acc,
            "val_loss":      val_loss,
            "val_acc":       val_acc,
            "val_macro_f1":  val_f1
        })

        print(f"[Fold{fold_id+1} Epoch{epoch+1}/{config.EPOCHS}] "
              f"Train Loss {train_loss:.4f} Acc {train_acc:.3f} | "
              f"Val Loss {val_loss:.4f} Acc {val_acc:.3f} F1 {val_f1:.4f}")

        # ---------- Checkpoint & Early Stopping ----------
        save_checkpoint({
            "epoch":      epoch,
            "state_dict": model.state_dict(),
            "optimizer":  optimizer.state_dict()
        }, is_best=(val_f1 > best_val_f1),
           filename=f"fold{fold_id+1}_epoch{epoch+1}.pt")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
        
    wandb.finish()


def predict_test(model_paths):
    loader = make_test_dataloader()
    preds = []
    for path in model_paths:
        model = build_model().to(config.DEVICE)
        load_checkpoint(model, filename=path)
        model.eval()
        all_out = []
        for imgs, _ in loader:
            imgs = imgs.to(config.DEVICE)
            with torch.no_grad():
                out = model(imgs)
            all_out.append(out.softmax(1).cpu().numpy())
        preds.append(np.concatenate(all_out, axis=0))

    avg_preds = np.mean(preds, axis=0)
    labels = avg_preds.argmax(axis=1)
    sub = pd.read_csv(config.SUBMISSION_CSV)
    sub["label"] = labels
    sub.to_csv("submission.csv", index=False)


def main():
    folds = make_kfold_dataloaders()
    for i, (tr, vl) in enumerate(folds):
        train_one_fold(i, tr, vl)
    # predict_test([...])


if __name__ == "__main__":
    main()