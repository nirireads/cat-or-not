import os
import numpy as np
from PIL import Image
import h5py

# Resize constant (keep consistent everywhere)
IMAGE_SIZE = 64  

# ------------------------------
# load Coursera-style H5 dataset
# ------------------------------
def load_h5_file(h5_path, x_key, y_key):
    with h5py.File(h5_path, "r") as f:
        X = np.array(f[x_key][:]).reshape(f[x_key].shape[0], -1).T / 255.0
        Y = np.array(f[y_key][:]).reshape(1, -1)
    return X, Y




