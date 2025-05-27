# inference.py
import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import timm
from PIL import Image
import pandas as pd
from tqdm import tqdm

import config

# Hyperparameters
CHECKPOINT_PATH = '/home/wfscontrol/rock_classifier/best_model.pt'
NUM_CLASSES = 7

# Class names (Alphabetical order)
class_names = [
    'Andesite',
    'Basalt',
    'Etc',
    'Gneiss',
    'Granite',
    'Mud_Sandstone',
    'Weathered_Rock'
]

# Load model and checkpoint
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=NUM_CLASSES)

# 체크포인트 로드 및 state_dict 추출
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
# 저장된 키 확인 (예: 'epoch', 'state_dict', 'optimizer')
if isinstance(checkpoint, dict):
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        # state_dict만 저장된 경우
        state_dict = checkpoint
else:
    # checkpoint 자체가 state_dict인 경우
    state_dict = checkpoint

# DataParallel prefix 제거 (필요 시)
clean_state = {}
for k, v in state_dict.items():
    new_key = k.replace('module.', '')
    clean_state[new_key] = v

model.load_state_dict(clean_state)
model = model.to(device)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
])

# Test data loading
test_df = pd.read_csv(config.TEST_CSV)
image_ids = test_df['ID'].astype(str).tolist()

# Predicting
preds = []
for image_id in tqdm(image_ids, desc='Predicting'):
    img_path = os.path.join(config.DATA_DIR, f"{image_id}.jpg")
    if not os.path.exists(img_path):
        print(f"Warning: Image not found: {img_path}")
        preds.append('Etc')
        continue
    image = Image.open(img_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        pred = output.argmax(dim=1).item()
        rock_name = class_names[pred]
        preds.append(rock_name)

# Making a submission DataFrame
submission = pd.DataFrame({
    'ID': image_ids,
    'rock_type': preds
})
submission.to_csv(config.SUBMISSION_CSV, index=False)
print(f"✅ 제출 파일 저장 완료: {config.SUBMISSION_CSV}")
