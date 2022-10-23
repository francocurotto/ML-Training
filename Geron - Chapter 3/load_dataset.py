import numpy as np

def load_dataset():
    X = np.load("datasets/data.npy", allow_pickle=True)
    y = np.load("datasets/target.npy", allow_pickle=True)
    y = y.astype(np.uint8)
    return X, y