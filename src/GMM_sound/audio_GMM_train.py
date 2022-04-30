from ikrlib import wav16khz2mfcc, train_gmm
import numpy as np
from numpy.random import randint
import sys, os
import pickle

# General parameters
SEGMENT = 200 # Size of the clipping window for silence removal
CUTOFF = 180 # Miliseconds to cut from the beginning of a record
ITERATIONS = 100 # Amount of iterations to train GMM
TARGET_CLUSTERS = 3 # Amount of clusters to model the target
OTHER_CLUSTERS = 50 # Amount of clusters to model the rest of the data

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
        subtract_means(sample[:,i])

    # For all the coeff collections, count the mean of zeroth coefficients
    mean_energy = np.mean(sample[:][:,0])

    # Remove segments which are quieter than mean value
    seg_list = []
    for seg in segment(sample, SEGMENT):
        if (np.mean(seg[:][:,0]) > mean_energy):
            seg_list.append(seg)

    return np.vstack(seg_list)

# Centers the coefficients' function around zero
def subtract_means(coeff_progress):
    mean = np.mean(coeff_progress)
    for i in range(0, len(coeff_progress)):
        coeff_progress[i] = coeff_progress[i] - mean

def throw_error():
    print("Use:\n 'python3 audio_GMM_train.py [TRAIN_TARGET_PATH] [TRAIN_NON_TARGET_PATH]'")
    exit()

def main():
    # Arg check
    if len(sys.argv) != 3 or not os.path.isdir(sys.argv[1]) or not os.path.isdir(sys.argv[2]):
        throw_error()

    # File reading
    print("Reading files...")
    blockPrint()

    # Creates a list of n (n = files.Count) matrices miliseconds x 13 coeffs
    train_t = list(wav16khz2mfcc(sys.argv[1]).values())
    train_n = list(wav16khz2mfcc(sys.argv[2]).values())

    enablePrint()
    print("Files read!")
    print("Preparing data...")

    # Data preparation
    train_tt = []
    for sample in train_t:
        train_tt.append(remove_silence(sample))

    train_nn = []
    for sample in train_n:
        train_nn.append(remove_silence(sample))

    # Slaps the matrices into one big matrix all miliseconds x 13 coeffs
    train_t = np.vstack(train_tt)
    train_n = np.vstack(train_nn)

    print("Data prepared!")

    print("Training GMM...")

    # Arbitrary amount
    Clusters_t = TARGET_CLUSTERS
    Clusters_n = OTHER_CLUSTERS

    # Creates a vector of n = Clusters vectors of 13 coeffs, picks random
    Means_t = train_t[randint(1, len(train_t), Clusters_t)]
    Means_n = train_n[randint(1, len(train_n), Clusters_n)]

    # Creates a list of n = Clusters covariance matrices 13x13
    Covs_t = [np.cov(train_t.T)] * Clusters_t
    Covs_n = [np.cov(train_n.T)] * Clusters_n

    # Creates a vector of n = Clusters 1/Clusters
    Ws_t = np.ones(Clusters_t) / Clusters_t
    Ws_n = np.ones(Clusters_n) / Clusters_n

    # The learning itself
    for jj in range(ITERATIONS):
    	Ws_t, Means_t, Covs_t, TTL_t = train_gmm(train_t, Ws_t, Means_t, Covs_t)
    	Ws_n, Means_n, Covs_n, TTL_n = train_gmm(train_n, Ws_n, Means_n, Covs_n)
    	print('Iteration: %d./%d' % ((jj + 1), ITERATIONS))

    print("GMM trained!")
    print("Generating new parameters...")

    with open('ws_t.pickle', 'wb') as f:
        pickle.dump(Ws_t, f)
    with open('means_t.pickle', 'wb') as f:
        pickle.dump(Means_t, f)
    with open('covs_t.pickle', 'wb') as f:
        pickle.dump(Covs_t, f)
    with open('ws_n.pickle', 'wb') as f:
        pickle.dump(Ws_n, f)
    with open('means_n.pickle', 'wb') as f:
        pickle.dump(Means_n, f)
    with open('covs_n.pickle', 'wb') as f:
        pickle.dump(Covs_n, f)

    print("Nex GMM parameters generated!")

if __name__ == "__main__":
    main()
