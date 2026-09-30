import gzip
import pickle
import math
import numpy as np
import matplotlib.pyplot as plt

# Constants
DATA_DIR_PATH = "../data"
IMG_DIR_PATH = "../Data_images"

### Data Input

def read_dataset(data_path = DATA_DIR_PATH):
    """
    Read in the input datasets, and return a tuple of:
        training | validation | testing

        train_x, train_y = training
        valid_x, valid_y = validation
        test_x, test_y = testing
    """

    with gzip.open(f"{data_path}/mnist.pkl.gz", "rb") as f:
        train_set, valid_set, test_set = pickle.load(f, encoding="latin1")

    return train_set, valid_set, test_set
    
### Data Output

def save_points(data_points, name="data_point"):
    """
    Saves the data point(s) in a square grayscale image file.
    """

    for idx, data_point in enumerate(data_points):
        img_w = int(math.sqrt(data_point.shape[0]))
        img = np.reshape(data_point, shape=(img_w, img_w))

        plt.imshow(img, cmap="gray", vmin=0, vmax=1)
        plt.axis("off")
        fig_name = f"{IMG_DIR_PATH}/{name}"
        if len(data_points) > 1:
            fig_name += f"-{idx}"
        plt.savefig(f"{fig_name}.png", bbox_inches="tight", pad_inches=0.1)
