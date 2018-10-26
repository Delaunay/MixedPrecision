#!/bin/sh

rsync -ax -e /data/lisa/data/ImageNet2012_jpeg/* /Tmp/delaunap/img_net/ &

DATA_LOCATION=/data/lisa/data/ImageNet2012_jpeg/
DATA_LOCATION=/Tmp/delaunap/img_net/
REPORT_NAME='report.csv'

ARGS="--gpu --report $REPORT_NAME --data $DATA_LOCATION --static-loss-scale 128 --prof 20"

BATCH_SIZE="128 256"
WORKERS="1 2 4 8 16 32 64"

declare -a CONFIG=("" "--half" "--use-dali" "--half --use-dali")

for batch in $BATCH_SIZE; do
    for worker in $WORKERS; do
        for arg_option in "${CONFIG[@]}"; do
            resnet-18-pt $ARGS -b $batch -j $worker $arg_option
            resnet-50-pt $ARGS -b $batch -j $worker $arg_option
        done
    done
done
