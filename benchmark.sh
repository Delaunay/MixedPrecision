#!/bin/bash

#rsync -ax -e /data/lisa/data/ImageNet2012_jpeg/* /Tmp/delaunap/img_net/ &

DATA_LOCATION=/data/lisa/data/ImageNet2012_256x256/train
DATA_LOCATION=/data/lisa/data/ImageNet2012_jpeg/train

DATA_LOCATION_BENZINA=/Tmp/delaunap/img_h264
DATA_LOCATION_BENZINA=/data/lisa/data/ImageNet2012_h264

# Local Copy
#DATA_LOCATION=/Tmp/delaunap/img_net/train

REPORT_NAME='report.csv'

BATCH_SIZE="256"
WORKERS="1 2 4"

CONFIG=("" --half)
LOADERS="torch dali"
seed=0

for batch in $BATCH_SIZE; do
    for worker in $WORKERS; do
        for arg_option in "${CONFIG[@]}"; do
            for loader in $LOADERS; do
                # -- Benzina
                if [ $loader = "benzina" ]; then
                    DATA_LOCATION=$DATA_LOCATION_BENZINA
                fi

                ARGS="--gpu --report $REPORT_NAME --data $DATA_LOCATION --static-loss-scale 128 --prof 10"

                #sudo sh -c 'echo 1 >/proc/sys/vm/drop_caches'
                #sudo sh -c 'echo 2 >/proc/sys/vm/drop_caches'
                #sudo sh -c 'echo 3 >/proc/sys/vm/drop_caches'
                free -th >> buffer_evolution.txt
                resnet-18-pt $ARGS --loader $loader -b $batch --seed $seed -j $worker $arg_option
                free -th >> buffer_evolution.txt
                seed=$(($seed + 1))
                sleep 5

                # Prevent the OS to cache stuff
                #sudo sh -c 'echo 1 >/proc/sys/vm/drop_caches'
                #sudo sh -c 'echo 2 >/proc/sys/vm/drop_caches'
                #sudo sh -c 'echo 3 >/proc/sys/vm/drop_caches'

                free -th >> buffer_evolution.txt
                resnet-50-pt $ARGS --loader $loader --seed $seed -b $batch -j $worker $arg_option
                free -th >> buffer_evolution.txt
                seed=$(($seed + 1))
                sleep 5

            done
        done
    done
done
