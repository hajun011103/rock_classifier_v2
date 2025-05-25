# config.py (추가 설정)
import os
import torch

# Path
DATA_DIR         = 'home/wfscontrol/Downloads/rocks'
TRAIN_DIR        = 'home/wfscontrol/Downloads/rocks/train'
TEST_DIR         = 'home/wfscontrol/Downloads/rocks/test'
SUBMISSION_CSV   = os.path.join(DATA_DIR, "sample_submission.csv")
TEST_CSV         = os.path.join(DATA_DIR, "test.csv")
SAVE_DIR         = 'home/wfscontrol/rock_classifier'

# Hyperparameters
BATCH_SIZE       = 16
NUM_CLASSES      = 7  # including 'etc'
N_FOLDS          = 5  # KFold 개수
EPOCHS           = 50
LR               = 1e-4
WEIGHT_DECAY     = 1e-2
PATIENCE         = 5

# WandB settings
PROJECT_NAME     = "rock_classification"
RUN_NAME         = "exp3"

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")