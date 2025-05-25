import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import pandas as pd
from tqdm import tqdm

# ----------------------
# 1. 설정
# ----------------------
CHECKPOINT_PATH = '/app/checkpoints/checkpoint_epoch1.pt'
TEST_IMG_DIR = '/data/test'
TEST_CSV_PATH = '/data/test.csv'
SUBMIT_CSV_PATH = 'submission.csv'
NUM_CLASSES = 7

# 클래스 이름 리스트 (알파벳 순서로)
class_names = [
    'Andesite',
    'Basalt',
    'Etc',
    'Gneiss',
    'Granite',
    'Mud_Sandstone',
    'Weathered_Rock'
]

# ----------------------
# 2. 장치 설정 및 모델 로드
# ----------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.convnext_tiny(weights=None)
model.classifier[2] = nn.Linear(model.classifier[2].in_features, NUM_CLASSES)

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

# ----------------------
# 3. 전처리 정의
# ----------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
])

# ----------------------
# 4. 테스트 데이터 로드
# ----------------------
test_df = pd.read_csv(TEST_CSV_PATH)
image_ids = test_df['ID'].astype(str).tolist()

# ----------------------
# 5. 추론
# ----------------------
preds = []
for image_id in tqdm(image_ids, desc='Predicting'):
    img_path = os.path.join(TEST_IMG_DIR, f"{image_id}.jpg")
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

# ----------------------
# 6. 제출 파일 생성
# ----------------------
submission = pd.DataFrame({
    'ID': image_ids,
    'rock_type': preds
})
submission.to_csv(SUBMIT_CSV_PATH, index=False)
print(f"✅ 제출 파일 저장 완료: {SUBMIT_CSV_PATH}")
