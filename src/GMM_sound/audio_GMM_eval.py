from ikrlib import wav16khz2mfcc, logpdf_gmm
from audio_GMM_train import blockPrint, enablePrint, remove_silence
import numpy as np
import sys, os
import pickle
from pathlib import Path

def throw_error():
    print("Use: \n'python3 audio_GMM_eval.py [EVAL_PATH]")
    exit()

def main():
    # Arg check
    if len(sys.argv) != 2 or not os.path.isdir(sys.argv[1]):
        throw_error()

    # File reading
    print("Reading files...")
    blockPrint()

    # A dictionary with the filename as key
    test = wav16khz2mfcc(sys.argv[1])

    enablePrint()
    print("Files read!")
    print("Preparing data...")

    test_v = []

    for sample in list(test.values()):
        test_v.append(remove_silence(sample))

    # Updates the original dictionaries' values with new edited data
    i = 0
    for k in test:
        test[k] = test_v[i]
        i +=1

    print("Data prepared!")

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

    # We expect 50% of real values to be targets
    Apr_t = 0.5
    Apr_n = 1 - Apr_t

    string = ''

    print("Classifying...")

    # Sum the likelihoods for each frame
    for tst in test:
        lhood_t = logpdf_gmm(test[tst], Ws_t, Means_t, Covs_t)
        lhood_n = logpdf_gmm(test[tst], Ws_n, Means_n, Covs_n)

        # Save the total as filename's value
        test[tst] = ((sum(lhood_t) + np.log(Apr_t)) - (sum(lhood_n) + np.log(Apr_n)))

    # Convert to <-1.0, 1.0> range and print
    OldPosRange = (max(test.values()) - 0)
    OldNegRange = (0 - min(test.values()))

    NewPosRange = 1.0
    NewNegRange = 1.0

    for tst in test:
        abs = 1 if test[tst] > 0 else 0

        if test[tst] > 0:
            test[tst] = ((test[tst]) * NewPosRange) / OldPosRange
        else:
            test[tst] = (((test[tst] - min(test.values())) * NewNegRange) / OldNegRange) + (-1.0)

        string += os.path.basename(Path(tst).stem) + " " + str("{:.5f}".format(test[tst])) + " " +  str(abs) + "\n"

    # Encode and print to file
    ascii = string.encode('ascii')
    with open('../../results_eval/audio_GMM.out', 'wb') as f:
        f.write(ascii)

    print("Classified and results written to file \"audio_GMM.out\"!")

if __name__ == "__main__":
    main()
