# config.py (추가 설정)
import os
import torch

# Path
DATA_DIR         = '/home/wfscontrol/Downloads/rocks'
TRAIN_DIR        = '/home/wfscontrol/Downloads/rocks/train'
TEST_DIR         = '/home/wfscontrol/Downloads/rocks/test'
SUBMISSION_CSV   = os.path.join(DATA_DIR, "sample_submission.csv")
TEST_CSV         = os.path.join(DATA_DIR, "test.csv")
SAVE_DIR         = '/home/wfscontrol/rock_classifier'

# Hyperparameters
BATCH_SIZE       = 16
NUM_CLASSES      = 7  
EPOCHS           = 50
LR               = 1e-4
WEIGHT_DECAY     = 1e-2
PATIENCE         = 5

# WandB settings
PROJECT_NAME     = "rock_classification"
RUN_NAME         = "exp3"
# wandb api key = "629278f83a01d6c995c7bf2ea8504729f8a1bca8"

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cross Validatoin
N_FOLDS          = 5
ETC_NAME         = "Etc"