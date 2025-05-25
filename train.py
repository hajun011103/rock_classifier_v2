# train.py (KFold 루프 + 테스트 예측) without FocalLoss and Scheduler
import wandb
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
import config
from dataset import make_kfold_dataloaders, make_test_dataloader
from model import build_model
from utils import save_checkpoint

def train_one_fold(fold_id, train_loader, val_loader):
    wandb.init(project=config.PROJECT_NAME,
               name=f"{config.RUN_NAME}_fold{fold_id+1}")

    model = build_model()
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=config.LR,
                                  weight_decay=config.WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()

    best_val_f1, patience = -1.0, 0
    hist = {"train_loss":[], "val_loss":[], "train_acc":[], "val_acc":[], "val_f1":[]}

    for epoch in range(config.EPOCHS):
        # ---------- Train ----------
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for imgs, targets in train_loader:
            imgs = imgs.to(config.DEVICE)
            labels = targets.argmax(dim=1).to(config.DEVICE)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                logits = model(imgs)
                loss   = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            tr_loss    += loss.item() * imgs.size(0)
            preds       = logits.argmax(dim=1)
            tr_correct += (preds == labels).sum().item()
            tr_total   += labels.size(0)

        train_loss = tr_loss / tr_total
        train_acc  = tr_correct / tr_total

        # ---------- Validate ----------
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        y_true, y_pred = [], []
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs = imgs.to(config.DEVICE)
                labels = targets.argmax(dim=1).to(config.DEVICE)
                with torch.cuda.amp.autocast():
                    logits = model(imgs)
                    loss   = criterion(logits, labels)
                val_loss   += loss.item() * imgs.size(0)
                preds       = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total   += labels.size(0)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        val_loss = val_loss / val_total
        val_acc  = val_correct / val_total
        val_f1   = f1_score(y_true, y_pred, average='macro')

        # ---------- Log to WandB ----------
        wandb.log({
            "epoch":          epoch + 1,
            "train_loss":     train_loss,
            "train_acc":      train_acc,
            "val_loss":       val_loss,
            "val_acc":        val_acc,
            "val_macro_f1":   val_f1
        })  # :contentReference[oaicite:0]{index=0}

        print(f"[Fold {fold_id+1} | Epoch {epoch+1}/{config.EPOCHS}]  "
              f"Train Loss {train_loss:.4f}  Acc {train_acc:.3f}  |  "
              f"Val Loss {val_loss:.4f}  Acc {val_acc:.3f}  |  "
              f"Macro-F1 {val_f1:.4f}")

        # ---------- Checkpoint & Early Stopping ----------
        save_checkpoint({
            "epoch":      epoch,
            "state_dict": model.state_dict(),
            "optimizer":  optimizer.state_dict()
        }, is_best=(val_f1 > best_val_f1),
           filename=f"fold{fold_id+1}_epoch{epoch+1}.pt")

        if val_f1 > best_val_f1:
            best_val_f1, patience = val_f1, 0
        else:
            patience += 1
            if patience >= config.PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

    wandb.finish()



def predict_test(model_paths):
    loader = make_test_dataloader()
    preds = []
    for path in model_paths:
        model = build_model()
        model.load_state_dict(torch.load(path)["state_dict"])
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
    # KFold 학습
    folds = make_kfold_dataloaders()
    for i, (tr, vl) in enumerate(folds):
        train_one_fold(i, tr, vl)
    # 테스트 예측 예시
    # predict_test(["./checkpoints/fold1_best.pt", ...])


if __name__ == "__main__":
    main()
