import numpy as np
import numpy.linalg as la

cimport cython
cimport numpy as np
from libc.math cimport abs, cos, sqrt, M_PI


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void normalize_columns(double[:, ::1] D):
    cdef int m = D.shape[0]
    cdef int n = D.shape[1]

    cdef int row, col
    cdef double acc

    for col in range(n):
        acc = 0.
        for row in range(m):
            acc += D[row, col]**2

        acc = sqrt(acc)

        for row in range(m):
            D[row, col] /= acc


@cython.boundscheck(False)
@cython.wraparound(False)
def construct_dictionary(double[:, ::1] D):
    cdef int m = D.shape[0]
    cdef int n = D.shape[1]

    cdef int row, col

    for row in range(m):
        D[row, 0] = n**(-.5)

    for col in range(1, n):
        for row in range(m):
            D[row, col] = (2. / n)**(.5) * cos(M_PI / n * (row + .5) * col)

    normalize_columns(D)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void mp(double[:] x,
             double[:, ::1] D,
             double[:] r_buf,
             double[:] s_buf,
             int iter_max,
             double global_eps):

    cdef int m = D.shape[0]
    cdef int n = D.shape[1]

    cdef int i, row, col
    cdef double acc

    cdef int w_argmax
    cdef double w_max

    for col in range(n):
        s_buf[col] = 0.

    if iter_max < 0:
        iter_max = n

    r_buf[:] = x
    for i in range(iter_max):
        w_max = 0.
        w_argmax = 0

        for col in range(n):
            acc = 0.
            for row in range(m):
                acc += D[row, col] * r_buf[row]

            if abs(acc) > abs(w_max):
                w_max = acc
                w_argmax = col

        s_buf[w_argmax] = w_max

        acc = 0.
        for row in range(m):
            r_buf[row] -= w_max * D[row, w_argmax]
            acc += r_buf[row]**2

        if sqrt(acc) < global_eps:
            break


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void omp(double[:] x,
              double[:, ::1] D,
              double[:] r_buf,
              double[:] s_buf,
              int iter_max,
              double global_eps):

    cdef int m = D.shape[0]
    cdef int n = D.shape[1]

    cdef double[:, ::1] D_new = np.zeros_like(D)

    cdef int i, row, col
    cdef double acc

    cdef int w_argmax
    cdef double w_max

    if iter_max < 0:
        iter_max = n

    r_buf[:] = x
    for i in range(iter_max):
        w_max = 0.
        w_argmax = 0

        for col in range(n):
            acc = 0.
            for row in range(m):
                acc += D[row, col] * r_buf[row]

            if abs(acc) > abs(w_max):
                w_max = acc
                w_argmax = col

        for row in range(m):
            D_new[row, w_argmax] = D[row, w_argmax]

        s = la.pinv(D_new) @ x

        for row in range(m):
            if s_buf[row] == 0.:
                r_buf[row] = x[row]
            else:
                acc = 0.
                for col in range(n):
                    acc += D_new[row, col] * s_buf[col]

                r_buf[row] = x[row] - acc

        acc = 0.
        for row in range(m):
            r_buf[row] -= w_max * D[row, w_argmax]
            acc += r_buf[row]**2

        if sqrt(acc) < global_eps:
            break


@cython.boundscheck(False)
@cython.wraparound(False)
def update_matrix(double[:, ::1] img,
                  int height,
                  int width,
                  int block_size,
                  double[:, ::1] sparse_dict,
                  double[:, ::1] sparse_mat,
                  double[:] activity_levels,
                  str mp_method,
                  int mp_iter_max,
                  double mp_global_eps):

    cdef int m = sparse_dict.shape[0]
    cdef int n = sparse_dict.shape[1]

    cdef double[:] vect = np.empty(m, dtype='double')
    cdef double[:] buf = np.empty(m, dtype='double')
    cdef double[:] sparse_vect = np.zeros(n, dtype='double')

    cdef int i, row, col, row_, col_
    cdef double activity_level

    cdef void (*approximate)(
        double[:], double[:, ::1], double[:], double[:], int, double)

    if mp_method == 'mp':
        approximate = mp
    elif mp_method == 'omp':
        approximate = omp
    else:
        raise ValueError("'method' must be either 'mp' or 'omp'")

    i = 0
    for row in range(height - block_size + 1):
        for col in range(width - block_size + 1):

            for row_ in range(block_size):
                for col_ in range(block_size):
                    vect[row_ * block_size + col_] = img[row + row_, col + col_]

            approximate(vect, sparse_dict, buf, sparse_vect,
                        mp_iter_max, mp_global_eps)

            activity_level = 0.
            for row_ in range(m):
                activity_level += abs(sparse_vect[row_])

            if activity_level > activity_levels[i]:
                for row_ in range(m):
                    sparse_mat[row_, i] = sparse_vect[row_]

                activity_levels[i] = activity_level

            i += 1


@cython.boundscheck(False)
@cython.wraparound(False)
def reconstruct_image(int height,
                      int width,
                      int block_size,
                      np.ndarray[np.double_t, ndim=2] sparse_dict,
                      np.ndarray[np.double_t, ndim=2] sparse_mat):

    cdef double[:, ::1] fused_image, fused_reconstruction

    cdef int i, row, col, row_, col_

    fused_image = np.zeros((height, width), dtype='double')
    fused_reconstruction = sparse_dict @ sparse_mat

    i = 0
    for row in range(height - block_size + 1):
        for col in range(width - block_size + 1):

            for row_ in range(block_size):
                for col_ in range(block_size):
                    fused_image[row + row_, col + col_] += \
                        fused_reconstruction[row_ * block_size + col_, i]
            i += 1

    return fused_image
