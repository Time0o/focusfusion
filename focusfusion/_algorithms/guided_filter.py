from typing import List, Tuple

import numpy as np

from .._common.filters import get_average_filter, \
                              get_gauss_filter, \
                              get_laplace_filter

from .._common.images import normalize


DEFAULT_LAPLACE_ALPHA = 0

DEFAULT_GAUSS_RADIUS = 5
DEFAULT_GAUSS_STDDEV = 5

DEFAULT_DECOMPOSE_AVERAGE_RADIUS = 10

DEFAULT_GUIDED_RADIUS_BASE = 45
DEFAULT_GUIDED_EPS_BASE = .3
DEFAULT_GUIDED_RADIUS_DETAIL = 7
DEFAULT_GUIDED_EPS_DETAIL = 1e-6


def decompose_details(img: np.ndarray,
                      average_radius: int) -> Tuple[np.ndarray, np.ndarray]:

    avg = get_average_filter(average_radius, img.shape)

    base = avg(img)
    detail = img - base

    return base, detail


def saliency_map(img: np.ndarray) -> np.ndarray:
    laplace = get_laplace_filter(DEFAULT_LAPLACE_ALPHA)
    gauss = get_gauss_filter(DEFAULT_GAUSS_RADIUS, DEFAULT_GAUSS_STDDEV)

    return gauss(np.abs(laplace(img)))


def weight_maps(images: List[np.ndarray]) -> List[np.ndarray]:
    saliency_maps = [saliency_map(img) for img in images]

    maxima = np.maximum.reduce(saliency_maps)

    return [(smap == maxima).astype(np.uint8) for smap in saliency_maps]


def guided_filter(img: np.ndarray,
                  guide: np.ndarray,
                  r: int,
                  eps: float) -> np.ndarray:

    squared = img * img
    cross = img * guide

    E = get_average_filter(r, img.shape)

    E_img, E_guide, E_squared, E_cross = map(E, [img, guide, squared, cross])

    a = (E_cross - E_img * E_guide) / (E_squared - E_img * E_img + eps)
    b = E_guide - a * E_img

    return E(a) * guide + E(b)


def smooth_weight_maps(images: List[np.ndarray],
                       guided_radius: int,
                       guided_eps: float) -> List[np.ndarray]:

    wmaps = weight_maps(images)

    for i in range(len(images)):
        wmap = guided_filter(images[i], wmaps[i], guided_radius, guided_eps)
        wmaps[i] = normalize(wmap)

    sums = np.add.reduce([wmap for wmap in wmaps])

    return [wmap / sums for wmap in wmaps]


def fuse_images(images: List[np.ndarray],
                decompose_average_radius: int = DEFAULT_DECOMPOSE_AVERAGE_RADIUS,
                guided_radius_base: int = DEFAULT_GUIDED_RADIUS_BASE,
                guided_eps_base: float = DEFAULT_GUIDED_EPS_BASE,
                guided_radius_detail: int = DEFAULT_GUIDED_RADIUS_DETAIL,
                guided_eps_detail: float = DEFAULT_GUIDED_EPS_BASE) -> np.ndarray:

    bases = []
    details = []
    for img in images:
        base, detail = decompose_details(img, decompose_average_radius)
        bases.append(base)
        details.append(detail)

    wmaps_bases = smooth_weight_maps(
        bases, guided_radius_base, guided_eps_base)

    wmaps_details = smooth_weight_maps(
        details, guided_radius_detail, guided_eps_detail)

    bases_fused = np.sum(np.array(wmaps_bases) * bases, axis=0)
    details_fused = np.sum(np.array(wmaps_details) * details, axis=0)

    return bases_fused + details_fused
