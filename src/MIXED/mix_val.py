from ikrlib import wav16khz2mfcc, logpdf_gmm
import numpy as np
import sys, os
import pickle
import albumentations as A
from torch import nn
from albumentations.pytorch import ToTensorV2
from PIL import Image

DEVICE = 'cuda'
PATH_TARGET = "../../SUR_projekt2021-2022/target_train/"
PATH_NONTARGET = "../../SUR_projekt2021-2022/non_target_train/"

PATH_TARGET_VAL = "../../SUR_projekt2021-2022/target_train/"
PATH_NONTARGET_VAL = "../../SUR_projekt2021-2022/non_target_train/"

TRANSFORMS_TEST = A.Compose([
    A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255, ),
    ToTensorV2(),
])


class CNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):
        super(CNN, self).__init__()
        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), \
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Flatten()

        )
        self.f1 = nn.Linear(1600, num_classes)

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        x = self.f1(x)
        x = nn.Sigmoid()(x)
        return x


# General parameters
SEGMENT = 200  # Size of the clipping window for silence removal
CUTOFF = 180  # Miliseconds to cut from the beginning of a record


# Silences printing output of ikrlib functions
def blockPrint():
    sys.stdout = open(os.devnull, 'w')


# Enables printing
def enablePrint():
    sys.stdout = sys.__stdout__


# Segments the single recording to parts of size seg_len
def segment(sample, seg_len):
    for i in range(0, len(sample), seg_len):
        yield sample[i:(i + seg_len)]


# Cuts off the initial two seconds and the silent parts of the recording and substracts means
def remove_silence(sample):
    # Initial pop cutoff
    sample = sample[CUTOFF:]

    # Subtract means of each coefficient progression to assure invariance
    for i in range(0, 12):
        subtract_means(sample[:, i])

    # For all the coeff collections, count the mean of zeroth coefficients
    mean_energy = np.mean(sample[:][:, 0])

    # Remove segments which are quieter than mean value
    seg_list = []
    for seg in segment(sample, SEGMENT):
        if (np.mean(seg[:][:, 0]) > mean_energy):
            seg_list.append(seg)

    return np.vstack(seg_list)


# Centers the coefficients' function around zero
def subtract_means(coeff_progress):
    mean = np.mean(coeff_progress)
    for i in range(0, len(coeff_progress)):
        coeff_progress[i] = coeff_progress[i] - mean


def throw_error():
    print("Use: \n'python3 audio_GMM_eval.py [EVAL_PATH]")
    exit()


def remove_silence_custom(input_segment):
    test_v = []
    for sample in list(input_segment.values()):
        test_v.append(remove_silence(sample))

    # Updates the original dictionaries' values with new edited data
    i = 0
    for k in input_segment:
        input_segment[k] = test_v[i]
        i += 1

    return input_segment


def load_gauss_model():
    with open('audio_GMM_model/ws_t.pickle', 'rb') as f:
        Ws_t = pickle.load(f)
    with open('audio_GMM_model/means_t.pickle', 'rb') as f:
        Means_t = pickle.load(f)
    with open('audio_GMM_model/covs_t.pickle', 'rb') as f:
        Covs_t = pickle.load(f)
    with open('audio_GMM_model/ws_n.pickle', 'rb') as f:
        Ws_n = pickle.load(f)
    with open('audio_GMM_model/means_n.pickle', 'rb') as f:
        Means_n = pickle.load(f)
    with open('audio_GMM_model/covs_n.pickle', 'rb') as f:
        Covs_n = pickle.load(f)

    return Ws_t, Means_t, Covs_t, Ws_n, Means_n, Covs_n


def evaluate_folder_cnn(model, PATH_TO_EVAL, is_target):
    results = {}
    for file in os.listdir(PATH_TO_EVAL):
        if file.endswith(".png"):
            pillow_image = Image.open(os.path.join(PATH_TO_EVAL, file))

            np_image = np.array(pillow_image)
            A_image = TRANSFORMS_TEST(image=np_image)["image"]
            device_A_image = A_image.unsqueeze(0).to(device=DEVICE)

            scores = model(device_A_image)
            scores = scores.tolist()[0]
            results[os.path.join(PATH_TO_EVAL, file)] = {"prediction_cnn": scores, "is_target": is_target}

    return results


def evaluate_data_GMM(input_segment, Ws_t, Means_t,
                      Covs_t, Ws_n, Means_n, Covs_n,
                      Apr_t, Apr_n):
    """
        for tst in audio_eval:
            lhood_t = logpdf_gmm(audio_eval[tst], Ws_t, Means_t, Covs_t)
            lhood_n = logpdf_gmm(audio_eval[tst], Ws_n, Means_n, Covs_n)

            # Save the total as filename's value
            audio_eval[tst] = ((sum(lhood_t) + np.log(Apr_t)) - (sum(lhood_n) + np.log(Apr_n)))

        # Convert to <-1.0, 1.0> range and print
        OldPosRange = (max(audio_eval.values()) - 0)
        OldNegRange = (0 - min(audio_eval.values()))

        for tst in audio_eval:
            if audio_eval[tst] > 0:
                audio_eval[tst] = ((audio_eval[tst]) * 1.0) / OldPosRange
            else:
                audio_eval[tst] = (((audio_eval[tst] - min(audio_eval.values())) * 1.0) / OldNegRange) + (-1.0)

        print(audio_eval)
        """

    for tst in input_segment:
        lhood_t = logpdf_gmm(input_segment[tst], Ws_t, Means_t, Covs_t)
        lhood_n = logpdf_gmm(input_segment[tst], Ws_n, Means_n, Covs_n)

        # Save the total as filename's value
        input_segment[tst] = ((sum(lhood_t) + np.log(Apr_t)) - (sum(lhood_n) + np.log(Apr_n)))

    # Convert to <-1.0, 1.0> range and print
    OldPosRange = (max(input_segment.values()) - 0)
    OldNegRange = (0 - min(input_segment.values()))

    for tst in input_segment:
        if input_segment[tst] > 0:
            input_segment[tst] = ((input_segment[tst]) * 1.0) / OldPosRange
        else:
            input_segment[tst] = (((input_segment[tst] - min(input_segment.values())) * 1.0) / OldNegRange) + (-1.0)

    return input_segment  # Dictionary ---> 'eval/eval_1371.wav': 0.10647536993240893


"""
{
    mena : {
        gmm,
        cnn,
        is_target,
        prediction,
        who_decide,
    }


}
"""


def main():
    # Arg check
    #  if len(sys.argv) != 2 or not os.path.isdir(sys.argv[1]):
    #     throw_error()

    # File reading
    # blockPrint()

    # A dictionary with the filename as key
    path_target_audio_train = wav16khz2mfcc(PATH_TARGET)
    path_nontarget_audio_train = wav16khz2mfcc(PATH_NONTARGET)
    path_target_audio_val = wav16khz2mfcc(PATH_TARGET_VAL)
    path_nontarget_audio_val = wav16khz2mfcc(PATH_NONTARGET_VAL)

    enablePrint()

    path_target_audio_train_removed_silence = remove_silence_custom(path_target_audio_train)
    path_nontarget_audio_train_removed_silence = remove_silence_custom(path_nontarget_audio_train)
    path_target_audio_val_removed_silence = remove_silence_custom(path_target_audio_val)
    path_nontarget_audio_val_removed_silence = remove_silence_custom(path_nontarget_audio_val)

    Apr_t = 0.5
    Apr_n = 1 - Apr_t

    Ws_t, Means_t, Covs_t, Ws_n, Means_n, Covs_n = load_gauss_model()

    path_target_audio_train_res = evaluate_data_GMM(path_target_audio_train_removed_silence,
                                                    Ws_t, Means_t,
                                                    Covs_t, Ws_n, Means_n, Covs_n,
                                                    Apr_t, Apr_n)

    path_nontarget_audio_train_res = evaluate_data_GMM(path_nontarget_audio_train_removed_silence,
                                                       Ws_t, Means_t,
                                                       Covs_t, Ws_n, Means_n, Covs_n,
                                                       Apr_t, Apr_n)

    path_target_audio_val_res = evaluate_data_GMM(path_target_audio_val_removed_silence,
                                                  Ws_t, Means_t,
                                                  Covs_t, Ws_n, Means_n, Covs_n,
                                                  Apr_t, Apr_n)

    path_nontarget_audio_val_res = evaluate_data_GMM(path_nontarget_audio_val_removed_silence,
                                                     Ws_t, Means_t,
                                                     Covs_t, Ws_n, Means_n, Covs_n,
                                                     Apr_t, Apr_n
                                                     )

    for keys in path_target_audio_train_res:
        prediction = path_target_audio_train_res[keys]
        path_target_audio_train_res[keys] = {"prediction_gmm": prediction, "is_target": 1}

    for keys in path_nontarget_audio_train_res:
        prediction = path_nontarget_audio_train_res[keys]
        path_nontarget_audio_train_res[keys] = {"prediction_gmm": prediction, "is_target": 0}

    for keys in path_target_audio_val_res:
        prediction = path_target_audio_val_res[keys]
        path_target_audio_val_res[keys] = {"prediction_gmm": prediction, "is_target": 1}

    for keys in path_nontarget_audio_val_res:
        prediction = path_nontarget_audio_val_res[keys]
        path_nontarget_audio_val_res[keys] = {"prediction_gmm": prediction, "is_target": 0}

    model_CNN = CNN().to(DEVICE)
    model_CNN.eval()
    cnn_results_target_train = evaluate_folder_cnn(model_CNN, PATH_TARGET, 1)
    cnn_results_non_target_train = evaluate_folder_cnn(model_CNN, PATH_NONTARGET, 0)
    cnn_results_target_val = evaluate_folder_cnn(model_CNN, PATH_TARGET_VAL, 1)
    cnn_results_non_target_val = evaluate_folder_cnn(model_CNN, PATH_NONTARGET_VAL, 0)

    def merge_two_dicts(x, y):
        z = x.copy()  # start with keys and values of x
        z.update(y)  # modifies z with keys and values of y
        return z

    cnn_fin = merge_two_dicts(cnn_results_target_train, cnn_results_non_target_train)
    cnn_fin = merge_two_dicts(cnn_fin, cnn_results_target_val)
    cnn_fin = merge_two_dicts(cnn_fin, cnn_results_non_target_val)

    gmm_fin = merge_two_dicts(path_target_audio_train_res, path_nontarget_audio_train_res)
    gmm_fin = merge_two_dicts(gmm_fin, path_target_audio_val_res)
    gmm_fin = merge_two_dicts(gmm_fin, path_nontarget_audio_val_res)

    results_fin = {}
    for key in cnn_fin:
        name = key[:-4]

        audio = name + ".wav"
        image = key
        prediction = 0
        who_predicted = ""
        if abs(gmm_fin[audio]["prediction_gmm"]) > abs(
                cnn_fin[image]["prediction_cnn"][0] - cnn_fin[image]["prediction_cnn"][1]):
            prediction = gmm_fin[audio]["prediction_gmm"]
            who_predicted = "gmm"
        else:
            who_predicted = "cnn"
            if cnn_fin[image]["prediction_cnn"][0] > cnn_fin[image]["prediction_cnn"][1]:
                prediction = cnn_fin[image]["prediction_cnn"][0]
            else:
                prediction = - cnn_fin[image]["prediction_cnn"][1]

        real_answer = 0
        if prediction > 0:
            real_answer = 1
        results_fin[name] = {"prediction_gmm": gmm_fin[audio]["prediction_gmm"],
                             "is_target_gmm": gmm_fin[audio]["is_target"],
                             "prediction_cnn": cnn_fin[image]["prediction_cnn"],
                             "is_target_cnn": cnn_fin[image]["is_target"],
                             "mixed_prediction": prediction,
                             "who_predict": who_predicted,
                             "real_answer": real_answer
                             }

    print(len(results_fin))
    false = 0
    miss = 0
    for key in results_fin:
        if results_fin[key]["is_target_gmm"] == 1:
            if results_fin[key]["real_answer"] != 1:
                miss += 1
        else:
            if results_fin[key]["real_answer"] == 1:
                false += 1
    print("total", len(results_fin), "false", false, "miss", miss, "percentage ",
          100 - (false + miss) / (len(results_fin) / 100))


if __name__ == "__main__":
    main()
