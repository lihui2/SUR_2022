from config import DATASET_TEST_PATH, \
    DATASET_NUM_WORKERS, DATASET_TRAIN_PATH, \
    NUM_EPOCHS, \
    DEVICE, \
    BATCH_SIZE, \
    LOAD_CHECK_POINT, \
    CHECK_POINT_PATH, \
    LEARNING_RATE, \
    CHECK_POINT_NAME, \
    SAVE_CHECK_POINT, \
    LOSS_FUNCTION, \
    TRANSFORMS_TRAIN, \
    TRANSFORMS_TEST

from dataset import ID_Dataset
from model import CNN
import torch
from torch import nn
from torch.utils.data import DataLoader
import os


def check_accuracy(loader: DataLoader, model: nn.Module, optimezer):
    model.eval()
    num_of_correct = 0
    num_of_samples = 0
    loss = 0
    differences = []
    losses = []
    for batch_idx, (x, y) in enumerate(loader):
        x = x.to(device=DEVICE)
        y = y.to(device=DEVICE)
        result = model(x)
        loss = loss_fn(result, y)
        losses.append(loss.item())
        # difference = abs(result[0][0] - y[0][0]).item()
        # losses.append( ( loss.item(), loss.item() < 0.5  ) )
        # differences.append( ( difference, difference < 0.5 ))
    loss_mean = 0
    for loss in losses:
        loss_mean += loss
    loss_mean = loss_mean / len(losses)

    print("Accuraccy  loss_mean on on test dataloader is ", loss_mean)
    model.train()


if __name__ == "__main__":

    print(f'Using {DEVICE} device')
    print("Dataset Path = ", DATASET_TEST_PATH)
    print("DATASET_NUM_WORKERS = ", DATASET_NUM_WORKERS)
    print("NUM_EPOCHS = ", NUM_EPOCHS)
    print("Loading dataset")

    training_data = ID_Dataset(DATASET_TRAIN_PATH, transforms=TRANSFORMS_TRAIN)
    test_data = ID_Dataset(DATASET_TEST_PATH, transforms=TRANSFORMS_TEST)
    print("Dataset loaded")

    train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=DATASET_NUM_WORKERS)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=DATASET_NUM_WORKERS)
    print("Dataloader loaded")

    model = CNN().to(DEVICE)
    loss_fn = LOSS_FUNCTION
    print("LOAD_CHECK_POINT ", LOAD_CHECK_POINT)
    if LOAD_CHECK_POINT:
        model.load_state_dict(torch.load(
            os.path.join(CHECK_POINT_PATH, CHECK_POINT_NAME)))
        model.eval()

    print("Model structure: ", model, "\n\n")
    optimezer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


    running_loss = 0.0

    for epoch in range(NUM_EPOCHS):
        for batch_idx, (data, targets) in enumerate(train_dataloader):
            data = data.to(device=DEVICE)
            targets = targets.to(device=DEVICE)
            scores = model(data)
            loss = loss_fn(scores, targets)

            optimezer.zero_grad()
            loss.backward()
            optimezer.step()
            running_loss += loss.item()

        print('EPOCH -> [%d] , loss: %.4f' % (epoch + 1, running_loss / NUM_EPOCHS))

        running_loss = 0.0
        if (epoch % 50 == 0 and epoch != 0):
            check_accuracy(test_dataloader, model, optimezer)

            if SAVE_CHECK_POINT:
                torch.save(model.state_dict(), os.path.join(CHECK_POINT_PATH, CHECK_POINT_NAME))

        if SAVE_CHECK_POINT:
            torch.save(model.state_dict(), os.path.join(CHECK_POINT_PATH, CHECK_POINT_NAME))

    check_accuracy(test_dataloader, model, optimezer)
