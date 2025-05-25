# train.py (KFold 루프 + 테스트 예측) without FocalLoss and Scheduler
import wandb
import torch
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
import config
from dataset import make_kfold_dataloaders, make_test_dataloader
from model import build_model
from utils import save_checkpoint

def train_one_fold(fold_id, train_loader, val_loader):
    wandb.init(project=config.PROJECT_NAME, name=f"{config.RUN_NAME}_fold{fold_id+1}")

    model = build_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()

    best_f1, patience = -1, 0
    for epoch in range(config.EPOCHS):
        # Training
        model.train()
        for imgs, targets in tqdm(train_loader, desc=f"Fold{fold_id+1} Train Epoch{epoch+1}"):
            imgs, targets = imgs.to(config.DEVICE), targets.to(config.DEVICE)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(imgs)
                # targets is one-hot, convert to class indices
                labels = targets.argmax(dim=1)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # Validation (F1 스코어 계산 로직 추가 필요)
        model.eval()
        # ...
        save_checkpoint({"state_dict": model.state_dict()}, is_best=False,
                        filename=f"fold{fold_id+1}_epoch{epoch+1}.pt")
        # early stop 로직

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
