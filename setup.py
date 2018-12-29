#!/usr/bin/env python3

import os

import numpy as np
from setuptools import find_packages, setup
from Cython.Build import cythonize


DISTNAME = 'focusfusion'
VERSION = '0.0.1'
DESCRIPTION = 'Multifocus image fusion algorithm collection'
URL = 'https://github.com/Time0o/focusfusion'
LICENSE = 'MIT'
AUTHOR = 'Timo Nicolai'
AUTHOR_EMAIL = 'timonicolai@arcor.de'

SETUP_DIR = os.path.dirname(os.path.abspath(__file__))
CYTHON_DIR = os.path.join(SETUP_DIR, 'focusfusion/_cython')


def get_requirements():
    requirements = os.path.join(SETUP_DIR, 'requirements.txt')

    with open(requirements, 'r') as f:
        return f.read().splitlines()


def get_cython_modules():
    return cythonize(os.path.join(CYTHON_DIR, '*.pyx'))


def get_include_dirs():
    return [np.get_include()]


def setup_package():
    setup(
        name=DISTNAME,
        version=VERSION,
        description=DESCRIPTION,
        url=URL,
        license=LICENSE,
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        packages=find_packages(),
        ext_modules=get_cython_modules(),
        include_dirs=get_include_dirs(),
        install_requires=get_requirements(),
        zip_safe=True
    )


if __name__ == '__main__':
    setup_package()
