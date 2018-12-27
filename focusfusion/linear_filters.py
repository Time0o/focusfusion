import numpy as np

from .common.filters import get_gauss_filter
from .cython.linear_filters import blur_error


FLOAT_DTYPE = np.float64

BLUR_ESTIMATE_STEPS = 25
BLUR_ESTIMATE_DELTA = .2
BLUR_ESTIMATE_RADIUS = 2
BLUR_ESTIMATE_THRESH = .1


def blur_estimate(img_focussed: np.ndarray,
                  img_unfocussed: np.ndarray,
                  steps: int = BLUR_ESTIMATE_STEPS,
                  delta: float = BLUR_ESTIMATE_DELTA,
                  radius: float = BLUR_ESTIMATE_RADIUS,
                  thresh: float = BLUR_ESTIMATE_THRESH) -> float:

    # convert images to floating point
    img_focussed = img_focussed.astype(FLOAT_DTYPE)
    img_unfocussed = img_unfocussed.astype(FLOAT_DTYPE)

    # pre-compute reference errors
    err_ref = blur_error(img_focussed, img_unfocussed, *img_focussed.shape, radius)

    # determine optimal blur radii
    optimal_blur_radii = np.empty_like(err_ref)
    optimal_blur_residuals = np.full_like(err_ref, np.inf)

    for k in range(1, steps + 1):
        # blur focussed image
        blur_radius = k * delta

        gauss = get_gauss_filter(radius, blur_radius / np.sqrt(2))
        img_blurred = gauss(img_focussed)

        # calculate normalized errors
        err = blur_error(img_blurred, img_unfocussed, *img_blurred.shape, radius)
        residuals = err / err_ref

        # update optima
        improved = np.where(residuals < optimal_blur_residuals)

        optimal_blur_radii[improved] = blur_radius
        optimal_blur_residuals[improved] = residuals[improved]

    # compute weighted sum of optimal blur radii
    weights = (1 - optimal_blur_residuals) / thresh
    res = np.sum(weights * optimal_blur_radii) / np.sum(weights)

    return res


def fft_gaussian(radius: int, blur_radius: float) -> np.ndarray:
    x = np.arange(-radius, radius + 1)
    y = np.exp(-(np.pi * blur_radius * x)**2)

    return y * y[:, np.newaxis]


def fft_filtered(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    # determine necessary frequency domain padding
    radius = (kernel.shape[0] - 1) // 2

    pad_rows = 2 * radius + 1 - img.shape[0]
    pad_top = pad_rows // 2
    pad_bottom = pad_rows - pad_top

    pad_cols = 2 * radius + 1 - img.shape[1]
    pad_left = pad_cols // 2
    pad_right = pad_cols - pad_left

    pad = ((pad_top, pad_bottom), (pad_left, pad_right))

    # compute FFT
    freq = np.fft.fftshift(np.fft.fft2(img))

    # pad result
    freq = np.pad(freq, pad, mode='constant')

    # apply filter
    freq *= kernel

    # remove padding
    freq = freq[pad_top:-pad_bottom, pad_left:-pad_right]

    return np.fft.ifftshift(freq)


def fuse_images(img_fg: np.ndarray,
                img_bg: np.ndarray,
                blur_radius_fg: float = .0,
                blur_radius_bg: float = .0) -> np.ndarray:

    assert img_fg.shape == img_bg.shape

    # estimate fore- and background blur radii
    br1 = blur_radius_fg
    br2 = blur_radius_bg
    br12 = blur_estimate(img_fg, img_bg)
    br21 = blur_estimate(img_bg, img_fg)

    # determine filter radius
    r = max(img_fg.shape) // 2

    # construct frequency domain filters
    psf1 = fft_gaussian(r, br1)
    psf2 = fft_gaussian(r, br2)
    psf12 = fft_gaussian(r, br12)
    psf21 = fft_gaussian(r, br21)

    # compute desired linear combination of frequency domain representations
    with np.errstate(divide='ignore', invalid='ignore'):
        alpha = (psf1 - psf2 * psf12) / (1 - psf12 * psf21)
        alpha[r, r] = (br12**2 + br2**2 - br1**2) / (br12**2 + br21**2)

        beta = (psf2 - psf1 * psf21) / (1 - psf12 * psf21)
        beta[r, r] = (br21**2 + br1**2 - br2**2) / (br12**2 + br21**2)

    filtered_fg = fft_filtered(img_fg, alpha)
    filtered_bg = fft_filtered(img_bg, beta)

    # return spatial domain reconstruction
    return np.real(np.fft.ifft2(filtered_fg + filtered_bg))
