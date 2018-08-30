#!/usr/bin/env python2
from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name='GroopM',
    version='2.0.0',
    author='Tim Lamberton',
    author_email='t.lamberton@uq.edu.au',
    packages=['groopm'],
    scripts=['bin/groopm2'],
    url='',
    license='LICENSE.txt',
    description='Metagenomic binning suite',
    long_description=open('README.txt').read(),
    install_requires=[
        "numpy >= 1.9.0",
        "scipy >= 0.17.0",
        "matplotlib >= 1.3.0",
        "tables >= 3.2.0"
    ],
    include_dirs = [np.get_include()],
    ext_modules = cythonize("groopm/stream_ext.pyx"), 
)
