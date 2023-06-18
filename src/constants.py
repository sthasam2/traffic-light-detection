import os

BASE_DIR = os.path.abspath(__file__).parent

TRAIN_DATASET_DIR = os.path.join(BASE_DIR, "train")
TRAIN_DATASET_DIR_GREEN = os.path.join(TRAIN_DATASET_DIR, "0_green")
TRAIN_DATASET_DIR_YELLOW = os.path.join(TRAIN_DATASET_DIR, "1_yellow")
TRAIN_DATASET_DIR_RED = os.path.join(TRAIN_DATASET_DIR, "2_red")
TRAIN_DATASET_DIR_NOT = os.path.join(TRAIN_DATASET_DIR, "3_not")

TEST_IMAGE_DIR = os.path.join(BASE_DIR, "test", "test_images")

TRAINED_MODEL_DIR = os.path.join(BASE_DIR, "trained_model")
TRAINED_MODEL_FILE = os.path.join(TRAINED_MODEL_DIR, "traffic_v1.h5")