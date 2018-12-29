from typing import Callable, List, Tuple

import numpy as np

from ._algorithms.guided_filter import fuse_images as fuse_guided_filter
from ._algorithms.guided_filter import DEFAULT_DECOMPOSE_AVERAGE_RADIUS, \
                                       DEFAULT_GUIDED_RADIUS_BASE, \
                                       DEFAULT_GUIDED_EPS_BASE, \
                                       DEFAULT_GUIDED_RADIUS_DETAIL, \
                                       DEFAULT_GUIDED_EPS_DETAIL

from ._algorithms.linear_filters import fuse_images as fuse_linear_filters
from ._algorithms.linear_filters import DEFAULT_BLUR_RADIUS_FG, \
                                        DEFAULT_BLUR_RADIUS_BG

from ._algorithms.sparse_repr import fuse_images as fuse_sparse_repr
from ._algorithms.sparse_repr import DEFAULT_BLOCK_SIZE, \
                                     DEFAULT_MP_METHOD, \
                                     DEFAULT_MP_ITER_MAX, \
                                     DEFAULT_MP_GLOBAL_EPS

from ._common.images import normalize


def focusfuse(images: List[np.ndarray], algorithm: str, **kwargs) -> np.ndarray:
    """Fuse a series of images depicting the same scene but with different
    in-focus regions into one all-in-focus image.

    This problem is commonly referred to as *multifocus image fusion* in the
    literature and a wide variety of non-trivial algorithms have been devised to
    solve it accurately and efficiently. This function is a wrapper around
    independent from-scratch implementations of several of the most influential
    (i.e. most cited) algorithms of this type, see the description of the
    `algorithm` parameter for a complete list.

    Parameters
    ----------
    images : sequence of ndarrays
        List of input images to be fused. All images must have the same
        dimensions and should depict the same scene with different in-focus
        regions. The images must also already be aligned. Note that not all
        fusion algorithms can work with color images and some operate on a fixed
        number of input images.

    algorithm : str
        Concrete fusion algorithm to use. Each algorithm accepts a list of
        grayscale images of arbitrary length unless stated otherwise. The
        behaviour of most algorithms can also be tuned with one or several
        optional parameters passed as keyword arguments and outlined under
        `**kwargs`. Accepted values for this parameter are:

        'guided_filter' (based on [1])
            Fuse an arbitrary number of grayscale images depicting the same
            scene with different in-focus regions by weighted summation of base
            and detail layers of the input images (with weights obtained via
            guided filtering of discretized saliency maps).

        'sparse_repr' (based on [2])
            Fuse an arbitrary number of grayscale images depicting the same
            scene with different in-focus regions by transforming them into
            sparse representation matrices using some variant of the *matching
            pursuit* algorithm and reconstructing an all-in-focus image from a
            combination of these matrices. Several optional parameters can be
            used to influence the runtime of the algorithm and the quality of
            the fusion result (see `**kwargs**).

        'linear_filters' (based on [3])
            Accepts exactly two grayscale input images. This algorithm assumes
            that the first input image depicts a scene with an out-of-focus
            background and the second depicts the same scene with an
            out-of-focus foreground. By default, an all-in-focus fused image
            will be constructed via linear filtering operations. Two optional
            parameters can be used to separately control the exact blur levels
            or fore- and background in the fused image (see `**kwargs).

    **kwargs:
        'guided_filter'
            decompose_average_radius : int (default is {DEFAULT_DECOMPOSE_AVERAGE_RADIUS})
                Radius of the averaging filter used during the composition of
                each input image into base and detail layer.
            guided_radius_base : int (default is {DEFAULT_GUIDED_RADIUS_BASE})
                Radius of the guided filter used to smooth the base layer weight
                maps.
            guided_eps_base : float (default is {DEFAULT_GUIDED_EPS_BASE})
                Regularization parameter of the guided filter used to smooth the
                base layer weight maps.
            guided_radius_detail : int (default is {DEFAULT_GUIDED_RADIUS_DETAIL})
                Radius of the guided filter used to smooth the detail layer
                weight maps.
            guided_eps_detail : float
                Regularization parameter of the guided filter used to smooth the
                detail layer weight maps.

        'sparse_repr'
            block_size : int (default is {DEFAULT_BLOCK_SIZE})
                Each input image is transformed into a sparse representation
                matrix by transforming each possible image patch of size
                `block_size`x`block_size` into a sparse representation vector
                using some variant of the *matching pursuit* algorithm. Larger
                blocks may yield better fusion results (but beware: runtime
                increases exponentially with `block_size`). In [1], the authors
                recommend a block size of 8x8.
            mp_method : either 'mp' or 'omp' (default is '{DEFAULT_MP_METHOD}')
                *Matching pursuit* variant to employ. If this parameter is set
                to 'omp', *orthogonal matching pursuit* is used (instead of the
                standard *matching pursuit* algorithm) which can improve fusion
                result quality (but note that this is currently extremely slow).
            mp_iter_max : int (default is {DEFAULT_MP_ITER_MAX})
                *Matching pursuit* iteration limit, negative values imply no
                iteration limit. Smaller (positive) values can improve runtime
                at the cost of fusion result quality.
            mp_global_eps : float (default is {DEFAULT_MP_GLOBAL_EPS})
                *Matching pursuit* error threshold. Larger values can improve
                runtime at the cost of fusion result quality.

        'linear_filters'
            blur_radius_fg : float (default is {DEFAULT_BLUR_RADIUS_FG})
                Controls foreground blur in fused image, a value of 0.0
                corresponds to an in-focus foreground.
            blur_radius_bg : float (default is {DEFAULT_BLUR_RADIUS_BG})
                Controls background blur in fused image, a value of 0.0
                corresponds to an in-focus foreground.

    Returns
    -------
    fused : ndarray
        Fused all-in-focus (except for some specific `algorithm`/`**kwargs`
        combinations) image. Regardless of the dtypes of the input images this
        array will be of dtype 'double'.

    Raises
    ------
        ValueError
            If any input image is neither a grayscale nor rgb image, the shapes
            of the input images do not match, the selected fusion algorithm is
            not recognized or the number or type of the input images is not
            appropriate for the selected fusion algorithm.

    References
    ----------
    .. [1] S. Li, X. Kang and J. Hu, *Image Fusion With Guided Filtering*, in
           *IEEE Transactions on Image Processing*, vol. 22, no. 7,
           pp. 2864-2875, July 2013.

    .. [2] B. Yang and S. Li, *Multifocus Image Fusion and Restoration With
           Sparse Representation*, in *IEEE Transactions on Instrumentation and
           Measurement*, vol. 59, no. 4, pp. 884-892, April 2010.

    .. [3] A. Kubota and K. Aizawa, *Reconstructing arbitrarily focused images
           from two differently focused images using linear filters*, in *IEEE
           Transactions on Image Processing*, vol. 14, no. 11, pp. 1848-1859,
           Nov. 2005.

    Examples
    --------
    >>> from focusfusion import fuse_images
    >>> from skimage.io import imread
    >>> img1 = imread('img1.png')
    >>> img2 = imread('img2.png')
    >>> img_fused = focusfuse([img1, img2], algorithm='sparse_repr')
    ...
    """

    def is_grayscale(img: np.ndarray) -> bool:
        return len(img.shape) == 2 or len(img.shape) == 3 and img.shape[2] == 1

    def is_rgb(img: np.ndarray) -> bool:
        return len(img.shape) == 3 and img.shape[2] == 3

    class ImageType:
        def __init__(self, shape: Tuple[int, int], rgb: bool):
            self.shape = shape
            self.rgb = rgb

    image_types = []

    for img in images:
        if is_grayscale(img):
            image_types.append(ImageType(img.shape, rgb=False))
        elif is_rgb(img):
            image_types.append(ImageType(img.shape, rgb=True))
        else:
            err = "at least one input image is neither grayscale nor rgb"
            raise ValueError(err)

    if len(set([it.shape for it in image_types])) > 1:
        err = "input image shape mismatch"
        raise ValueError(err)

    class FusionAlgorithm:
        def __init__(self,
                     function: Callable,
                     num_inputs: int = None,
                     grayscale_only: bool = True,
                     rgb_only: bool = False):

            self.function = function
            self.num_inputs = num_inputs
            self.grayscale_only = grayscale_only
            self.rgb_only = rgb_only

    algo = {
        'guided_filter': FusionAlgorithm(fuse_guided_filter),
        'linear_filters': FusionAlgorithm(fuse_linear_filters, num_inputs=2),
        'sparse_repr': FusionAlgorithm(fuse_sparse_repr),
    }.get(algorithm)

    if algo is None:
        err = "unrecognized fusion algorithm: '{}'"
        raise ValueError(err.format(algorithm))

    if algo.num_inputs is not None and len(images) != algo.num_inputs:
        err = "'{}' expects {} input images but {} were provided"
        raise ValueError(err.format(algorithm, algo.num_inputs, len(images)))

    if algo.grayscale_only and any([it.rgb for it in image_types]):
        err = "'{}' expects grayscale only input images but one or more rgb images were provided"
        raise ValueError(err.format(algorithm))

    if algo.rgb_only and not all([it.rgb for it in image_types]):
        err = "'{}' expects rgb only input images but one or more grayscale images were provided"
        raise ValueError(err.format(algorithm))

    # normalize input images
    images = [normalize(img) for img in images]

    # run fusion algorithm
    if algo.num_inputs is None:
        res = algo.function(images, **kwargs)
    else:
        res = algo.function(*images, **kwargs)

    # make sure result is a numpy array
    res = np.array(res, dtype='double')

    # normalize result to range [0, 1]
    res -= res.min()
    res /= res.max()

    return res


focusfuse.__doc__ = focusfuse.__doc__.format(
    DEFAULT_BLUR_RADIUS_FG=DEFAULT_BLUR_RADIUS_FG,
    DEFAULT_BLUR_RADIUS_BG=DEFAULT_BLUR_RADIUS_BG,
    DEFAULT_BLOCK_SIZE=DEFAULT_BLOCK_SIZE,
    DEFAULT_MP_METHOD=DEFAULT_MP_METHOD,
    DEFAULT_MP_ITER_MAX=DEFAULT_MP_ITER_MAX,
    DEFAULT_MP_GLOBAL_EPS=DEFAULT_MP_GLOBAL_EPS,
    DEFAULT_DECOMPOSE_AVERAGE_RADIUS=DEFAULT_DECOMPOSE_AVERAGE_RADIUS,
    DEFAULT_GUIDED_EPS_BASE=DEFAULT_GUIDED_EPS_BASE,
    DEFAULT_GUIDED_EPS_DETAIL=DEFAULT_GUIDED_EPS_DETAIL,
    DEFAULT_GUIDED_RADIUS_BASE=DEFAULT_GUIDED_RADIUS_BASE,
    DEFAULT_GUIDED_RADIUS_DETAIL=DEFAULT_GUIDED_RADIUS_DETAIL)
