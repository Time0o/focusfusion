from typing import Callable, Tuple

import numpy as np
from numpy.lib.stride_tricks import as_strided


def apply_filter(img: np.ndarray,
                 kernel: np.ndarray,
                 pad_mode: str = 'constant') -> np.ndarray:

    pad_width = ((kernel.shape[0] - 1) // 2,
                 (kernel.shape[1] - 1) // 2)

    img = np.pad(img, pad_width, pad_mode)

    strided_shape = kernel.shape + (img.shape[0] - kernel.shape[0] + 1,
                                    img.shape[1] - kernel.shape[1] + 1)

    strided = as_strided(img, strided_shape, 2 * img.strides)

    return np.einsum('ij,ijkl->kl', kernel, strided)


def apply_separable_filter(img: np.ndarray,
                           kernel: np.ndarray,
                           pad_mode: str = 'constant') -> np.ndarray:

    img = np.pad(img, (kernel.shape[0] - 1) // 2, pad_mode)

    def convolve(arr):
        return np.convolve(arr, kernel, mode='valid')

    for axis in 0, 1:
        img = np.apply_along_axis(convolve, axis, img)

    return img


def get_average_filter(
    r: int,
    shape: Tuple[int, int],
    pad_mode: str = 'constant'

) -> Callable[[np.ndarray], np.ndarray]:

    kernel = np.ones(2 * r + 1, dtype='int')
    counts = box_filter(np.ones(shape, dtype='int'), kernel)

    return lambda img: apply_separable_filter(img, kernel, pad_mode) / counts


def get_gauss_filter(
    r: int,
    sigma: float,
    pad_mode: str = 'reflect'

) -> Callable[[np.ndarray], np.ndarray]:

    x = np.arange(-r, r + 1, dtype='double')

    kernel = np.exp(-(x * x) / (2 * sigma * sigma))
    kernel /= kernel.sum()

    return lambda img: apply_separable_filter(img, kernel, pad_mode)


def get_laplace_filter(
    alpha: float,
    pad_mode: str = 'reflect'

) -> Callable[[np.ndarray], np.ndarray]:

    a = alpha / 4
    b = (1 - alpha) / 4

    kernel = 4 / (alpha + 1) * np.array([[a, b, a], [b, -1, b], [a, b, a]])

    return lambda img: apply_filter(img, kernel, pad_mode)
