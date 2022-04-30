# This is a sample Python script.
import torch, os
from torch.utils.data import DataLoader
from src.NN_img_regression.model import CNN
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from src.NN_img_regression.config import DEVICE, CHECK_POINT_PATH, TRANSFORMS_TEST
from src.NN_img_regression.dataset import ID_Dataset

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    test_data = ID_Dataset(os.path.join("../../dataset", "dev", "data_img"), transforms=TRANSFORMS_TEST)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True, num_workers=1)

    model = CNN().to(DEVICE)
    model.load_state_dict(torch.load(os.path.join("..", "NN_img", CHECK_POINT_PATH, "model_img.pth")))
    model.eval()
    results = []
    for i in range(50):
        for batch_idx, (data, targets) in enumerate(test_dataloader):
            data = data.to(device=DEVICE)
            scores = model(data)
            # image = data.squeeze(0).permute(1, 2, 0)
            pil_image = transforms.ToPILImage()(data.squeeze(0))
            scores = scores.item()
            targets = targets.item()
            results.append((pil_image, scores, targets))


    def display_multiple_img(images, rows=1, cols=1):
        figure, ax = plt.subplots(nrows=rows, ncols=cols)
        for ind, values in enumerate(images):
            print(values)
            image, score, target = values

            if( ind  == rows * cols):
                break
            else:
                ax.ravel()[ind].imshow(image)
                title = "{:.3f}".format(score)+"\n"+"{:.3f}".format(target)
                ax.ravel()[ind].set_title(title)
                ax.ravel()[ind].set_axis_off()

        plt.tight_layout()
        plt.show()

    display_multiple_img(results, 5, 5)


