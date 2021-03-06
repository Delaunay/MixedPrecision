#!/usr/bin/env python

from setuptools import setup


if __name__ == '__main__':
    setup(
        name='MixedPrecision',
        version='0.0.0',
        description='Implement a few model to try out the new Tensor Core capabilities (F16)',
        author='Pierre Delaunay',
        packages=[
            'MixedPrecision',
            'MixedPrecision.pytorch',
            'MixedPrecision.tensorflow',
            'MixedPrecision.tools'],
        entry_points={
            'console_scripts': [
                # Pytorch
                'mnist-conv = MixedPrecision.pytorch.mnist_conv:main',
                'mnist-full = MixedPrecision.pytorch.mnist_fully_connected:main',
                'convnets = MixedPrecision.pytorch.convnet:main',

                'resnet-18 = MixedPrecision.pytorch.resnet:resnet18_main',
                'resnet-50 = MixedPrecision.pytorch.resnet:resnet50_main',
                'resnet-101 = MixedPrecision.pytorch.resnet:resnet101_main',
                'resnet-34 = MixedPrecision.pytorch.resnet:resnet34_main',
                'densenet-161 = MixedPrecision.pytorch.resnet:densenet161_main',
                #'resnet-50 = MixedPrecision.pytorch.resnet:resnet50_main',

                'mnist-conv-pt = MixedPrecision.pytorch.mnist_conv:main',
                'mnist-full-pt = MixedPrecision.pytorch.mnist_fully_connected:main',
                'resnet-18-pt = MixedPrecision.pytorch.resnet:resnet18_main',
                'resnet-50-pt = MixedPrecision.pytorch.resnet:resnet50_main',

                # Tensorflow
                'mnist-full-tf = MixedPrecision.tensorflow.mnist_fully_connected:main',

                # Data Loader Benchmark
                'benchloader = MixedPrecision.tools.loaders:main',

                # 
                'imgnet_preprocessor = MixedPrecision.pytorch.preprocessor:main',
                'hdf5_converter = MixedPrecision.tools.hdf5:main'
            ]
        }
    )
