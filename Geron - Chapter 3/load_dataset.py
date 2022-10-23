import numpy as np

def load_dataset():
    X = np.load("datasets/data.npy", allow_pickle=True)
    y = np.load("datasets/target.npy", allow_pickle=True)
    return X, y