import os
from PIL import Image
import numpy as np
import torch
from NN_img_classification.model import CNN
from NN_img_classification.config import TRANSFORMS_TEST,\
                                             DEVICE, \
                                             CHECK_POINT_PATH



PATH_TO_EVAL = "eval/"

if __name__ == "__main__":
    data = []
    model = CNN().to(DEVICE)
    model.load_state_dict(torch.load(os.path.join("src", "", "final_model", "900model_img.pth")))
    model.eval()

    for file in os.listdir(PATH_TO_EVAL):
        if file.endswith(".png"):
            pillow_image = Image.open(os.path.join(PATH_TO_EVAL,file))
            np_image = np.array(pillow_image)
            A_image = TRANSFORMS_TEST(image=np_image)["image"]
            device_A_image = A_image.unsqueeze(0).to(device=DEVICE)
            scores = model(device_A_image)
            scores = scores.tolist()[0]
            decision = 0
            results = []
            results.append(file.split('.')[0])
            if scores[0] > scores[1]:
                decision = 1
                results.append(str(scores[0]))
                results.append(str(decision))
            else:
                decision = 0
                results.append(str(scores[1]))
                results.append(str(decision))

            results_str = ' '.join(results)
            data.append(results_str)
    file = open("results_eval/nn_classification2.txt", "w")
    for line in data:
        file.write(line + "\n")
    file.close()
