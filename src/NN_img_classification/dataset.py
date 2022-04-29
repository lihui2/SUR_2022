import os

import cv2
import numpy as np
import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2
import glob
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from src.NN_img_classification.config import DATASET_TRAIN_PATH, TRANSFORMS_TRAIN,TARGET_MULTIPLY
import torchvision.transforms as T
transform = T.ToPILImage()

# convert the tensor to PIL image using above transform

TARGET = "target"
NON_TARGET = "non_target"



class ID_Dataset_train(Dataset):
    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms
        self.picture = []
        self.score = []
        folder_dir = root_dir

        for images in glob.iglob(f'{folder_dir}/{TARGET}/*'):
            if images.endswith(".png"):
                pillow_image = Image.open(os.path.abspath(images))
                np_image = np.array(pillow_image)
                for i in range(TARGET_MULTIPLY):
                    self.picture.append(np_image)
                    self.score.append([1.0, 0.0])

        for images in glob.iglob(f'{folder_dir}/{NON_TARGET}/*'):
            if images.endswith(".png"):
                pillow_image = Image.open(os.path.abspath(images))
                np_image = np.array(pillow_image)
                self.picture.append(np_image)
                self.score.append([0.0, 1.0])


    def __getitem__(self, id):
        x = self.picture[id]
        y = self.score[id]
        if self.transforms != None:
            x = self.transforms(image=x)["image"]
            y = torch.from_numpy(np.array(y))

        return x, y

    def __len__(self):
        return len(self.picture)



class ID_Dataset_test(Dataset):
    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms
        self.picture = []
        self.score = []
        folder_dir = root_dir

        for images in glob.iglob(f'{folder_dir}/{TARGET}/*'):
            if images.endswith(".png"):
                pillow_image = Image.open(os.path.abspath(images))
                np_image = np.array(pillow_image)
                #img = cv2.imread(os.path.abspath(images))
                self.picture.append(np_image)
                self.score.append([1.0, 0.0])

        for images in glob.iglob(f'{folder_dir}/{NON_TARGET}/*'):
            if images.endswith(".png"):
                pillow_image = Image.open(os.path.abspath(images))
                np_image = np.array(pillow_image)
                #img = cv2.imread(os.path.abspath(images))
                self.picture.append(np_image)
                self.score.append([0.0, 1.0])


    def __getitem__(self, id):
        x = self.picture[id]
        y = self.score[id]
        if self.transforms != None:
            x = self.transforms(image=x)["image"]
            y = torch.from_numpy(np.array(y))

        return x, y

    def __len__(self):
        return len(self.picture)


if __name__ == "__main__":
    train_data = ID_Dataset_train(DATASET_TRAIN_PATH, transforms=TRANSFORMS_TRAIN)
    # test_data = ID_Dataset(DATASET_TEST_PATH)
    print(train_data.picture)

    for i in range(10):
        train_img, train_score = train_data[i]
        T.ToPILImage()(train_img).show()


        print(train_img)
        print(train_img.size())
        print(train_score)
        print(train_score.size())
        exit(1)