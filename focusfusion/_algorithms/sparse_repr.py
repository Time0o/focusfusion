from typing import List

import numpy as np

from .._cython.sparse_repr import construct_dictionary, \
                                  reconstruct_image, \
                                  update_matrix


DEFAULT_BLOCK_SIZE = 4

DEFAULT_MP_METHOD = 'mp'
DEFAULT_MP_ITER_MAX = -1
DEFAULT_MP_GLOBAL_EPS = .1


def fuse_images(images: List[np.ndarray],
                block_size: int = DEFAULT_BLOCK_SIZE,
                mp_method: str = DEFAULT_MP_METHOD,
                mp_iter_max: int = DEFAULT_MP_ITER_MAX,
                mp_global_eps: float = DEFAULT_MP_GLOBAL_EPS):

    num_patches = np.prod((np.array(images[0].shape) - block_size + 1))
    patch_size = block_size**2

    sparse_dict = np.empty((patch_size, patch_size), dtype='double')
    construct_dictionary(sparse_dict)

    sparse_mat = np.empty((patch_size, num_patches), dtype='double')

    sparse_levels = np.zeros(num_patches, dtype='double')

    for img in images:
        update_matrix(img.astype('double'),
                      *images[0].shape, block_size,
                      sparse_dict, sparse_mat, sparse_levels,
                      mp_method, mp_iter_max, mp_global_eps)

    return reconstruct_image(*images[0].shape, block_size,
                             sparse_dict, sparse_mat)
