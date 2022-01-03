from setuptools import setup, find_packages

setup(
    name='depth_util',
    version='1.0.0',
    description='Utilities for conversion between depth and disparity images',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scikit-image',
        'matplotlib'
    ]
)
