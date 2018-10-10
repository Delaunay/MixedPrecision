#!/bin/sh

export  PYTORCH_RESULTS=./results/pytorch/

mkdir -p $PYTORCH_RESULTS
mkdir -p ./results/tensorflow/


nvprof --log-file $PYTORCH_RESULTS/mnist_full_single.csv --csv mnist-full --epochs 1 --data /data/ --hidden_size 16384 --gpu > $PYTORCH_RESULTS/mnist_full_single.out
nvprof --log-file $PYTORCH_RESULTS/mnist_full_half.csv --csv mnist-full --epochs 1 --data /data/ --hidden_size 16384 --gpu --half > $PYTORCH_RESULTS/mnist_full_half.out

nvprof --log-file $PYTORCH_RESULTS/mnist_conv_single.csv --csv mnist-conv --epochs 1 --data /data/ --hidden_size 1024 --conv_num 512 --gpu > $PYTORCH_RESULTS/mnist_conv_single.out
nvprof --log-file $PYTORCH_RESULTS/mnist_conv_half.csv --csv mnist-conv --epochs 1 --data /data/ --hidden_size 1024 --conv_num 512 --gpu --half > $PYTORCH_RESULTS/mnist_conv_half.out

nvprof --csv --log-file $PYTORCH_RESULTS/resnet18_single.csv resnet-18 --data /data/img_net/ --epochs 1 --gpu > $PYTORCH_RESULTS/resnet.out
nvprof --csv --log-file $PYTORCH_RESULTS/resnet18_half.csv resnet-18 --data /data/img_net/ --epochs 1 --gpu --half > $PYTORCH_RESULTS/resnet.out