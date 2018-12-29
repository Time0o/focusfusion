from itertools import product

import numpy as np


BLOCKDIM = 8
BLOCKNORM = 1 / BLOCKDIM**2

TINY_ELEM = 1e-14

DEFAULT_DECISION_THRESHOLD = 1e-3


def dct_matrix() -> np.ndarray:
    n = BLOCKDIM

    res = np.empty((n, n), dtype=np.float64)

    res[0, :] = 1 / np.sqrt(n)

    s = np.sqrt(2 / n)
    for i in range(1, n):
        for j in range(n):
            res[i, j] = s * np.cos((np.pi * (2 * j + 1) * i) / (2 * n))

    return res


def dct_pad(img: np.ndarray) -> np.ndarray:
    pad_rows = img.shape[0] % BLOCKDIM
    pad_cols = img.shape[1] % BLOCKDIM

    return np.pad(img, ((0, pad_rows), (0, pad_cols)), 'constant')


def spatial_freq_matrix_constant(C: np.ndarray) -> np.ndarray:
    b = np.zeros((BLOCKDIM, BLOCKDIM), dtype='double')

    i, j = np.indices(b.shape)
    b[(i == j) & (i < BLOCKDIM - 1)] = -1
    b[i == j + 1] = 1

    B = C @ b @ C.T

    D = B @ B.T
    D[D < TINY_ELEM] = 0

    E = np.empty((BLOCKDIM, BLOCKDIM), dtype='double')

    for i in range(BLOCKDIM):
        E[i, i] = D[i, i]

        for j in range(0, i):
            E[i, j] = D[i, i] + D[j, j]

    return E + E.T


def spatial_freq(b: np.ndarray, C: np.ndarray, E: np.ndarray) -> 'double':
    return BLOCKNORM * np.sum(E * np.square(C @ b @ C.T))


def create_decision_matrix(img_a: np.ndarray,
                           img_b: np.ndarray,
                           decision_threshold: 'double') -> np.ndarray:

    if not create_decision_matrix.init:
        C = dct_matrix()

        create_decision_matrix.C = C
        create_decision_matrix.E = spatial_freq_matrix_constant(C)

        create_decision_matrix.init = True

    img_a_padded = dct_pad(img_a)
    img_b_padded = dct_pad(img_b)

    W = np.empty((img_a_padded.shape[0] // BLOCKDIM,
                  img_a_padded.shape[1] // BLOCKDIM), dtype=np.int8)

    for r in range(W.shape[0]):
        for c in range(W.shape[1]):
            r_block = r * BLOCKDIM
            r_block_end = r_block + BLOCKDIM

            c_block = c * BLOCKDIM
            c_block_end = c_block + BLOCKDIM

            block_a = img_a[r_block:r_block_end,
                            c_block:c_block_end]

            block_b = img_b[r_block:r_block_end,
                            c_block:c_block_end]

            sf_a = spatial_freq(block_a,
                                create_decision_matrix.C,
                                create_decision_matrix.E)

            sf_b = spatial_freq(block_b,
                                create_decision_matrix.C,
                                create_decision_matrix.E)

            if sf_a > sf_b + decision_threshold:
                W[r, c] = 1
            elif sf_a < sf_b - decision_threshold:
                W[r, c] = -1
            else:
                W[r, c] = 0

    return W

create_decision_matrix.init = False


def improve_decision_matrix(W: np.ndarray) -> np.ndarray:
    R = np.zeros_like(W)

    W = np.pad(W, 1, 'constant')

    for r in range(R.shape[0]):
        for c in range(R.shape[1]):
            for offs_r, offs_c in product((-1, 0, 1), (-1, 0, 1)):
                R[r, c] += W[r + offs_r, c + offs_c]

    return R


def fuse_images(img_a: np.ndarray,
                img_b: np.ndarray,
                decision_threshold: 'double' = DEFAULT_DECISION_THRESHOLD) -> np.ndarray:

    W = create_decision_matrix(img_a, img_b, decision_threshold)
    R = improve_decision_matrix(W)

    img_fused = np.empty_like(img_a)

    for r in range(R.shape[0]):
        for c in range(R.shape[1]):
            r_block = r * BLOCKDIM
            r_block_end = r_block + BLOCKDIM

            c_block = c * BLOCKDIM
            c_block_end = c_block + BLOCKDIM

            tmp = R[r, c]

            if tmp > 0:
                block = img_a[r_block:r_block_end,
                              c_block:c_block_end]
            elif tmp < 0:
                block = img_b[r_block:r_block_end,
                              c_block:c_block_end]
            else:
                block_a = img_a[r_block:r_block_end,
                                c_block:c_block_end]

                block_b = img_b[r_block:r_block_end,
                                c_block:c_block_end]

                block = (block_a + block_b) / 2

            img_fused[r_block:r_block_end, c_block:c_block_end] = block

    return img_fused
