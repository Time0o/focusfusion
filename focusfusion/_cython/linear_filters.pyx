import numpy as np

cimport cython
cimport numpy as np


@cython.boundscheck(False)
@cython.wraparound(False)
def blur_error(double[:, ::1] img_fg,
               double[:, ::1] img_bg,
               int radius):

    cdef int m = img_fg.shape[0]
    cdef int n = img_fg.shape[0]

    # allocate accumulator images
    cdef double[:, ::1] diff = np.empty((m, n),
                                        dtype='double')

    cdef double[:, ::1] err = np.empty((m - 2 * radius, n - 2 * radius),
                                       dtype='double')

    # compute squared difference between input images
    cdef int row, col

    for row in range(m):
        for col in range(n):
            diff[row, col] = (img_fg[row, col] - img_bg[row, col])**2


    # sum differences in neightbourhood of each pixel
    cdef int r, row_, col_
    cdef double acc, eps = np.finfo('double').eps

    r = radius
    for row in range(r, m - r):
        for col in range(r, n - r):

            acc = 0.
            for row_ in range(row - r, row + r + 1):
                for col_ in range(col - r, col + r + 1):
                    acc += diff[row_, col_]

            err[row - r, col - r] = eps if acc == 0. else acc

    return np.asarray(err)
