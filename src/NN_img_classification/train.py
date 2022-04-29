from src.NN_img_classification.config import DATASET_TEST_PATH, \
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
import matplotlib.pyplot as plt
from dataset import ID_Dataset_train,ID_Dataset_test
from model import CNN
import torch
from torch import nn
from torch.utils.data import DataLoader
import os


def check_accuracy(loader: DataLoader, model: nn.Module, optimezer):
    model.eval()
    num_of_correct = 0
    false_alaram = 0
    miss = 0
    num_of_samples = 0
    loss = 0
    differences = []
    losses = []
    for batch_idx, (x, y) in enumerate(loader):
        x = x.to(device=DEVICE)
        y = y.to(device=DEVICE)
        result = model(x)
        loss = loss_fn(result, y)

        y_values = y.tolist()
        result = result.tolist()

        y_a, y_b = y_values[0]
        t_a, t_b = result[0]
        if y_a > y_b:
            if t_a > t_b:
                num_of_correct += 1
            else:
                false_alaram += 1
        if y_a < y_b:

            if t_a < t_b:
                num_of_correct += 1
            else:
                miss +=1
        num_of_samples += 1

        losses.append(loss.item())
        # difference = abs(result[0][0] - y[0][0]).item()
        # losses.append( ( loss.item(), loss.item() < 0.5  ) )
        # differences.append( ( difference, difference < 0.5 ))
    """    
    num_of_correct = 0
    num_of_samples = 0 
    for tup in losses:
        if tup[1] == True:
            num_of_correct +=1
        num_of_samples += 1
    print("Accuraccy loss on  test dataloader is ", num_of_correct/num_of_samples, " from ",num_of_samples ," are correct ", num_of_correct, "." )
    num_of_correct = 0
    num_of_samples = 0 
    for tup in differences:
        if tup[1] == True:
            num_of_correct +=1
        num_of_samples += 1
    print("Accuraccy diff on on test dataloader is ", num_of_correct/num_of_samples, " from ",num_of_samples ," are correct ", num_of_correct,  "." )
    """
    loss_mean = 0
    for loss in losses:
        loss_mean += loss
    loss_mean = loss_mean / len(losses)

    #print("Accuraccy  loss_mean on on test dataloader is ", loss_mean)
    print("Correct predictions is ", num_of_correct , "from ", num_of_samples," ",num_of_correct/num_of_samples,"%" )
    print("False alarm", false_alaram,"Miss", miss)
    model.train()
    return num_of_correct, num_of_samples, false_alaram, miss


if __name__ == "__main__":

    print(f'Using {DEVICE} device')
    print("Dataset Path = ", DATASET_TEST_PATH)
    print("DATASET_NUM_WORKERS = ", DATASET_NUM_WORKERS)
    print("NUM_EPOCHS = ", NUM_EPOCHS)
    print("Loading dataset")

    training_data = ID_Dataset_train(DATASET_TRAIN_PATH, transforms=TRANSFORMS_TEST)
    test_data = ID_Dataset_test(DATASET_TEST_PATH, transforms=TRANSFORMS_TEST)
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

    """
    for name, param in model.named_parameters():
        print(
            f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
    """
    running_loss = 0.0
    num_of_correct_array = []
    num_of_samples_array = []
    false_alarams = []
    misses = []
    epochs = []
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

        print('EPOCH -> [%d] , loss: %.10f' % (epoch + 1, running_loss / NUM_EPOCHS))

        running_loss = 0.0
        if (epoch % 20 == 0 ):
            num_of_correct, num_of_samples, false_alaram, miss = check_accuracy(test_dataloader, model, optimezer)
            num_of_correct_array.append(num_of_correct)
            num_of_samples_array.append(num_of_samples)
            false_alarams.append(false_alaram)
            misses.append(miss)
            epochs.append(epoch)

            if SAVE_CHECK_POINT:
                torch.save(model.state_dict(), os.path.join(CHECK_POINT_PATH, str(epoch)+".".join(CHECK_POINT_NAME.split("."))))

    num_of_correct, num_of_samples, false_alaram, miss  = check_accuracy(test_dataloader, model, optimezer)
    num_of_correct_array.append(num_of_correct)
    num_of_samples_array.append(num_of_samples)
    false_alarams.append(false_alaram)
    misses.append(miss)
    epochs.append(NUM_EPOCHS)
    if SAVE_CHECK_POINT:
        torch.save(model.state_dict(), os.path.join(CHECK_POINT_PATH, str(epoch)+".".join(CHECK_POINT_NAME.split("."))))

    fig, axs = plt.subplots(2, 1)
    correctness = axs[0].plot(epochs, [num_of_correct_array[x]/num_of_samples_array[0] for x in range(len(num_of_correct_array))])
    axs[0].legend(['num_of_correct/num_of_samples'],shadow=True, fancybox=True)

    #axs[0].set_xlim(0, 2)
    axs[0].set_xlabel('epochs')
    axs[0].set_ylabel('Correct %')

    false_alarams_l,misses_l = axs[1].plot(epochs, false_alarams, epochs,misses)
    axs[1].set_ylabel('Count')
    axs[1].set_xlabel('epochs')
    axs[1].legend((false_alarams_l, misses_l), ('false_alarms','misses'),shadow=True, fancybox=True)
    fig.tight_layout()
    plt.savefig('training.png')
    plt.show()
    """
    if SAVE_CHECK_POINT:
        torch.save(model.state_dict(), os.path.join(CHECK_POINT_PATH, CHECK_POINT_NAME))
    """