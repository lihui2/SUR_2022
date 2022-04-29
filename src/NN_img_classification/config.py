import os.path
import cv2
from torch import cuda
from torch import nn
from os import path
import albumentations as A
from albumentations.pytorch import ToTensorV2
IMAGE_SIZE = 80
scale = 1

DATASET_TRAIN_PATH = os.path.join("..", "..", "dataset", "train", "data_img")
DATASET_TEST_PATH = os.path.join("..", "..", "dataset", "dev", "data_img")


TARGET_MULTIPLY = 10
TRANSFORMS_TRAIN = A.Compose([

    A.PadIfNeeded(
       min_height=int(IMAGE_SIZE * scale),
        min_width=int(IMAGE_SIZE * scale),
        border_mode=cv2.BORDER_CONSTANT,
    ),
    A.RandomCrop(width=IMAGE_SIZE, height=IMAGE_SIZE),
    A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=0.3),
    A.ShiftScaleRotate(rotate_limit=15, p=0.3, border_mode=cv2.BORDER_CONSTANT),
    A.HorizontalFlip(p=0.5),
    A.CLAHE(p=0.4),
    A.Posterize(p=0.4),
    A.ToGray(p=0.4),
    A.ChannelShuffle(p=0.05),
    A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255, ),
    ToTensorV2(),
])

TRANSFORMS_TEST = A.Compose([
    A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255, ),
    ToTensorV2(),
])

DATASET_NUM_WORKERS = 2
NUM_EPOCHS = 1000
DEVICE = 'cuda' if cuda.is_available() else 'cpu'
BATCH_SIZE = 1
LOAD_CHECK_POINT = False
SAVE_CHECK_POINT = True
CHECK_POINT_PATH = os.path.join("models")
CHECK_POINT_NAME = "model_img.pth"
LEARNING_RATE = 1e-5
LOSS_FUNCTION = nn.CrossEntropyLoss()
