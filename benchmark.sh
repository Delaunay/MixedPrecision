#!/bin/sh

DATA_LOCATION=/Tmp/delaunap/img_net/
ARGS="--gpu --data $DATA_LOCATION --static-loss-scale 128 --prof 10"

BATCH_SIZE="128 256"
WORKERS="1 2 4 8 16"

for batch in BATCH_SIZE do
    for worker in WORKERS do
        resnet-18-pt $ARGS -b $batch -j $worker
        resnet-18-pt $ARGS -b $batch -j $worker --half

        resnet-50-pt $ARGS -b $batch -j $worker
        resnet-50-pt $ARGS -b $batch -j $worker --half
    done
done
