    
    
    
    
    resnet-50-pt --gpu --data ../data/imgnet/ImageNet2012_jpeg/ -b 128 -j 4 --static-loss-scale 128 --prof 10
    [   1][  10] Batch Time (avg: 0.3708, sd: 0.0000) Speed (avg: 345.2209) Data (avg: 0.0011, sd: 0.0006)
    [   1][  20] Batch Time (avg: 0.3690, sd: 0.0008) Speed (avg: 346.9080) Data (avg: 0.0010, sd: 0.0004)
    [   1][  30] Batch Time (avg: 0.3687, sd: 0.0007) Speed (avg: 347.1751) Data (avg: 0.0011, sd: 0.0004)
    [   1][  40] Batch Time (avg: 0.3686, sd: 0.0009) Speed (avg: 347.2632) Data (avg: 0.0011, sd: 0.0004)
    [   1][  50] Batch Time (avg: 0.3684, sd: 0.0009) Speed (avg: 347.4515) Data (avg: 0.0010, sd: 0.0004)
    [   1][  60] Batch Time (avg: 0.3683, sd: 0.0008) Speed (avg: 347.5011) Data (avg: 0.0010, sd: 0.0004)
    [   1][  70] Batch Time (avg: 0.3682, sd: 0.0008) Speed (avg: 347.5981) Data (avg: 0.0010, sd: 0.0003)
    [   1][  80] Batch Time (avg: 0.3682, sd: 0.0008) Speed (avg: 347.6247) Data (avg: 0.0009, sd: 0.0003)
    [   1][  90] Batch Time (avg: 0.3682, sd: 0.0007) Speed (avg: 347.6441) Data (avg: 0.0009, sd: 0.0003)
    [   1][ 100] Batch Time (avg: 0.3682, sd: 0.0007) Speed (avg: 347.6666) Data (avg: 0.0009, sd: 0.0003)
    |           Metric |  Average | Deviation |      Min |      Max |
    |-----------------:|---------:|----------:|---------:|---------:|
    | CPU Data loading |   0.0123 |    0.1221 |   0.0001 |   1.2392 |
    | GPU Data Loading |   0.0051 |    0.0004 |   0.0034 |   0.0070 |
    | Waiting for data |   0.0009 |    0.0003 |   0.0004 |   0.0037 |
    | CPU Compute Time |   0.3682 |    0.0007 |   0.3670 |   0.3722 |
    | GPU Compute Time |   0.3680 |    0.0006 |   0.3669 |   0.3707 |
    |  Full Batch Time |   0.3691 |    0.0009 |   0.3675 |   0.3735 |
    |    Compute Speed | 347.6666 |        NA | 343.9024 | 348.7361 |
    |  Effective Speed | 346.8003 |        NA | 342.7191 | 348.3067 |
    
