<p align="center">
  <img src="preview.png">
</p>

# Fuse Out of Focus Images

## Introduction

This Python package implements four different algorithms for fusing a series of
out of focus images into one sharp image. The following algorithms are supported:

| Algorithm         | Published In |
|-------------------|--------------|
| guided_filter     | [1]          |
| sparse_repr       | [2]          |
| dct_spatial_freq  | [3]          |
| linear_filters    | [4]          |

## Usage

Simply run `pip install .` to install the package, then images can be fused via:

```
from focusfusion import focusfuse
from skimage.io import imread

img1 = imread('img1.png')
img2 = imread('img2.png')

img_fused = focusfuse([img1, img2], algorithm='sparse_repr')
```

Where `sparse_repr` can be replaced by any of the other available algorithms.
See `help(focusfusion)` for a more detailed description and possible fine
tuning options. The Jupyter notebook under `demo` contains some visual examples.

## References

[1] S. Li, X. Kang and J. Hu, *Image Fusion With Guided Filtering*, in
    *IEEE Transactions on Image Processing*, vol. 22, no. 7,
    pp. 2864-2875, July 2013.

[2] B. Yang and S. Li, *Multifocus Image Fusion and Restoration With
    Sparse Representation*, in *IEEE Transactions on Instrumentation and
    Measurement*, vol. 59, no. 4, pp. 884-892, April 2010.

[3] L. Cao, L. Jin, H. Tao, G. Li, Z. Zhuang and Y. Zhang.,
    *Multi-Focus Image Fusion Based on Spatial Frequency in Discrete
    Cosine Transform Domain*, in *IEEE Signal Processing Letters*,
    vol. 22, no. 2, pp. 220-224, Feb. 2015.

[4] A. Kubota and K. Aizawa, *Reconstructing arbitrarily focused images
    from two differently focused images using linear filters*, in *IEEE
    Transactions on Image Processing*, vol. 14, no. 11, pp. 1848-1859,
    Nov. 2005.
