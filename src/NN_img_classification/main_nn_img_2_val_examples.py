# This is a sample Python script.
import torch, os
from torch.utils.data import DataLoader
from NN_img_classification.model import CNN
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from NN_img_classification.config import DEVICE, CHECK_POINT_PATH, TRANSFORMS_TEST
from NN_img_classification.dataset import ID_Dataset_test

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    test_data = ID_Dataset_test(os.path.join("../../dataset", "dev", "data_img"), transforms=TRANSFORMS_TEST)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True, num_workers=1)

    model = CNN().to(DEVICE)
    model.load_state_dict(torch.load(os.path.join("..", "NN_img_classification", CHECK_POINT_PATH, "900model_img.pth")))
    model.eval()
    results = []
    for i in range(50):
        for batch_idx, (data, targets) in enumerate(test_dataloader):
            data = data.to(device=DEVICE)
            scores = model(data)
            # image = data.squeeze(0).permute(1, 2, 0)
            pil_image = transforms.ToPILImage()(data.squeeze(0))
            scores = scores.tolist()
            targets = targets.tolist()
            results.append((pil_image, scores, targets))


    def display_multiple_img(images, rows=1, cols=1):
        figure, ax = plt.subplots(nrows=rows, ncols=cols)
        for ind, values in enumerate(images):
            image, scores, targets = values

            if( ind  == rows * cols):
                break
            else:
                ax.ravel()[ind].imshow(image)
                scores = ["{:1.3f}".format(score) for score in scores[0]]
                targets = ["{:1.3f}".format(target) for target in targets[0]]
                title = ' '.join(scores) + "\n" + ' '.join(targets)
                #title = ' '.join(targets)
                ax.ravel()[ind].set_title(title)
                ax.ravel()[ind].set_axis_off()

        plt.tight_layout()
        plt.show()

    display_multiple_img(results, 5, 5)

    """
    fig, ax = plt.subplots(1, 20)
    for result_index in range(len(results)):
        image, score , targets = results[result_index]
        ax[1,result_index].imshow(image)
        #ax[result_index].set_title("Score ", str(score),"Real ",str(targets))
    plt.show()  
    """
