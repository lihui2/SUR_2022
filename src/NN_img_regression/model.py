import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from src.NN_img_regression.config import TRANSFORMS_TRAIN, TRANSFORMS_TEST, DATASET_TRAIN_PATH, BATCH_SIZE, DATASET_NUM_WORKERS
from src.NN_img_regression.dataset import ID_Dataset


class CNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super(CNN, self).__init__()
        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), \
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), \
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), \
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),


        )
        self.fc1 = nn.Linear(1600, num_classes)

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        #x = nn.Sigmoid()(x)
        return x


if __name__ == "__main__":
    print("Test neural network")
    training_data = ID_Dataset(DATASET_TRAIN_PATH, transforms=TRANSFORMS_TRAIN)
    train_dataloader = DataLoader(training_data,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True,
                                  num_workers=DATASET_NUM_WORKERS)
    examples = iter(train_dataloader)
    example_data, example_targets = examples.next()

    # print("Input ", example_data)
    # print("Output ",example_targets)
    print("Input size ", example_data.size(), "Output size ", example_targets.size())
    model = CNN()
    result = model(example_data)
    print("result ", result, "target", example_targets)
    print("result ", result.size(), "target", example_targets.size())
