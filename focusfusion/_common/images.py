import numpy as np


def normalize(img: np.ndarray) -> np.ndarray:
    img = img.astype('double')

    img -= img.min()
    img /= img.max()

    return img
