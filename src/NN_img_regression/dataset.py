import os
import numpy as np
import torch
import glob
from PIL import Image
from torch.utils.data import Dataset
from src.NN_img_regression.config import DATASET_TRAIN_PATH, TRANSFORMS_TRAIN
import torchvision.transforms as T
transform = T.ToPILImage()


TARGET = "target"
NON_TARGET = "non_target"

class ID_Dataset(Dataset):
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
                self.picture.append(np_image)
                self.score.append([1])

        for images in glob.iglob(f'{folder_dir}/{NON_TARGET}/*'):
            if images.endswith(".png"):
                pillow_image = Image.open(os.path.abspath(images))
                np_image = np.array(pillow_image)
                self.picture.append(np_image)
                self.score.append([0])

        self.score = torch.FloatTensor(self.score)

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
    train_data = ID_Dataset(DATASET_TRAIN_PATH, transforms=TRANSFORMS_TRAIN)
    for i in range(10):
        train_img, train_score = train_data[i]
        T.ToPILImage()(train_img).show()
        print(train_img)
        print(train_img.size())
        print(train_score)
        print(train_score.size())
        exit(1)
