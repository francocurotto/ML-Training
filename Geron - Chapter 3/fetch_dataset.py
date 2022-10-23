import numpy as np
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist["data"], mnist["target"]
np.save("datasets/data.npy", X)
np.save("datasets/target.npy", y)