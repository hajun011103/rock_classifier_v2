# config.py
import os
import torch

# Path
DATA_DIR         = '/home/wfscontrol/Downloads/rocks'
TRAIN_DIR        = '/home/wfscontrol/Downloads/rocks/train'
TEST_DIR         = '/home/wfscontrol/Downloads/rocks/test'
SUBMISSION_CSV   = os.path.join(DATA_DIR, "sample_submission.csv")
TEST_CSV         = os.path.join(DATA_DIR, "test.csv")
SAVE_DIR         = '/home/wfscontrol/rock_classifier/rock_classifier_v2'

# Hyperparameters
BATCH_SIZE       = 80
NUM_CLASSES      = 7  
EPOCHS           = 50
LR               = 3.125e-5
MIN_LR           = 1e-6
WEIGHT_DECAY     = 1e-2
PATIENCE         = 5
WARM_UP_EPOCHS   = 5
SEED             = 42

# Tensorboard settings
RUN_NAME         = "rock_classifier"
# wandb api key = "629278f83a01d6c995c7bf2ea8504729f8a1bca8"

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cross Validatoin
N_FOLDS          = 7

# Cross Entropy Loss with Class Weights
class_counts = torch.tensor([43801, 26809, 15934, 73913, 92922, 89466, 37168], dtype=torch.float)

total = class_counts.sum()  # 380013
num_classes = len(class_counts)  # 7
weights = total / (num_classes * class_counts)

weights = weights.clamp(min=0.5, max=3.0) # 너무 극단적일때 클램프 (0.6, 3.4) 로 조정가능
# print("Class Weights:", weights) 

WEIGHTS = weights.to(DEVICE)