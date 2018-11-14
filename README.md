Mixed Precision
===============


Conv Net benchmarking utility.
Generate a report after 100 batches on the speed of the different stage of the training pipeline.

Example:

    resnet-18-pt --data /Tmp/delaunap/img_net --gpu -j 4 --loader torch -b 256 --prof 10
                              data: /Tmp/delaunap/img_net/
                           workers: 4
                       hidden_size: 64
                        hidden_num: 1
                       kernel_size: 3
                          conv_num: 32
                            epochs: 10
                        batch_size: 256
                                lr: 0.1
                          momentum: 0.9
                      weight_decay: 0.0001
                        print_freq: 10
                              half: False
                               gpu: True
                 static_loss_scale: 1
                dynamic_loss_scale: False
                              prof: None
                             shape: [1, 28, 28]
              log_device_placement: False
                     no_bench_mode: False
                       batch_reuse: False
                            report: None
                            warmup: False
                            loader: torch
                             async: False
                              seed: 0
                         GPU Count: 1
                          GPU Name: Tesla V100-PCIE-16GB
    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Conv2d-1        [256, 64, 112, 112]           9,408
           BatchNorm2d-2        [256, 64, 112, 112]             128
                  ReLU-3        [256, 64, 112, 112]               0
             MaxPool2d-4          [256, 64, 56, 56]               0
                Conv2d-5          [256, 64, 56, 56]          36,864
           BatchNorm2d-6          [256, 64, 56, 56]             128
                  ReLU-7          [256, 64, 56, 56]               0
                Conv2d-8          [256, 64, 56, 56]          36,864
           BatchNorm2d-9          [256, 64, 56, 56]             128
                 ReLU-10          [256, 64, 56, 56]               0
           BasicBlock-11          [256, 64, 56, 56]               0
               Conv2d-12          [256, 64, 56, 56]          36,864
          BatchNorm2d-13          [256, 64, 56, 56]             128
                 ReLU-14          [256, 64, 56, 56]               0
               Conv2d-15          [256, 64, 56, 56]          36,864
          BatchNorm2d-16          [256, 64, 56, 56]             128
                 ReLU-17          [256, 64, 56, 56]               0
           BasicBlock-18          [256, 64, 56, 56]               0
               Conv2d-19         [256, 128, 28, 28]          73,728
          BatchNorm2d-20         [256, 128, 28, 28]             256
                 ReLU-21         [256, 128, 28, 28]               0
               Conv2d-22         [256, 128, 28, 28]         147,456
          BatchNorm2d-23         [256, 128, 28, 28]             256
               Conv2d-24         [256, 128, 28, 28]           8,192
          BatchNorm2d-25         [256, 128, 28, 28]             256
                 ReLU-26         [256, 128, 28, 28]               0
           BasicBlock-27         [256, 128, 28, 28]               0
               Conv2d-28         [256, 128, 28, 28]         147,456
          BatchNorm2d-29         [256, 128, 28, 28]             256
                 ReLU-30         [256, 128, 28, 28]               0
               Conv2d-31         [256, 128, 28, 28]         147,456
          BatchNorm2d-32         [256, 128, 28, 28]             256
                 ReLU-33         [256, 128, 28, 28]               0
           BasicBlock-34         [256, 128, 28, 28]               0
               Conv2d-35         [256, 256, 14, 14]         294,912
          BatchNorm2d-36         [256, 256, 14, 14]             512
                 ReLU-37         [256, 256, 14, 14]               0
               Conv2d-38         [256, 256, 14, 14]         589,824
          BatchNorm2d-39         [256, 256, 14, 14]             512
               Conv2d-40         [256, 256, 14, 14]          32,768
          BatchNorm2d-41         [256, 256, 14, 14]             512
                 ReLU-42         [256, 256, 14, 14]               0
           BasicBlock-43         [256, 256, 14, 14]               0
               Conv2d-44         [256, 256, 14, 14]         589,824
          BatchNorm2d-45         [256, 256, 14, 14]             512
                 ReLU-46         [256, 256, 14, 14]               0
               Conv2d-47         [256, 256, 14, 14]         589,824
          BatchNorm2d-48         [256, 256, 14, 14]             512
                 ReLU-49         [256, 256, 14, 14]               0
           BasicBlock-50         [256, 256, 14, 14]               0
               Conv2d-51           [256, 512, 7, 7]       1,179,648
          BatchNorm2d-52           [256, 512, 7, 7]           1,024
                 ReLU-53           [256, 512, 7, 7]               0
               Conv2d-54           [256, 512, 7, 7]       2,359,296
          BatchNorm2d-55           [256, 512, 7, 7]           1,024
               Conv2d-56           [256, 512, 7, 7]         131,072
          BatchNorm2d-57           [256, 512, 7, 7]           1,024
                 ReLU-58           [256, 512, 7, 7]               0
           BasicBlock-59           [256, 512, 7, 7]               0
               Conv2d-60           [256, 512, 7, 7]       2,359,296
          BatchNorm2d-61           [256, 512, 7, 7]           1,024
                 ReLU-62           [256, 512, 7, 7]               0
               Conv2d-63           [256, 512, 7, 7]       2,359,296
          BatchNorm2d-64           [256, 512, 7, 7]           1,024
                 ReLU-65           [256, 512, 7, 7]               0
           BasicBlock-66           [256, 512, 7, 7]               0
            AvgPool2d-67           [256, 512, 1, 1]               0
               Linear-68                [256, 1000]         513,000
    ================================================================
    Total params: 11,689,512
    Trainable params: 11,689,512
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 147.00
    Forward/backward pass size (MB): 16074.95
    Params size (MB): 44.59
    Estimated Total Size (MB): 16266.55
    ----------------------------------------------------------------
    [   1][  10] Batch Time (avg: 0.2565, sd: 0.0000) Speed (avg: 998.1434) Data (avg: 7.7139, sd: 0.0000)
    [   1][  20] Batch Time (avg: 0.2519, sd: 0.0021) Speed (avg: 1016.1350) Data (avg: 1.3090, sd: 2.6183)
    [   1][  30] Batch Time (avg: 0.2524, sd: 0.0024) Speed (avg: 1014.3584) Data (avg: 1.5345, sd: 2.5739)
    [   1][  40] Batch Time (avg: 0.2524, sd: 0.0026) Speed (avg: 1014.2191) Data (avg: 1.4393, sd: 2.5644)
    [   1][  50] Batch Time (avg: 0.2524, sd: 0.0028) Speed (avg: 1014.4265) Data (avg: 1.2298, sd: 2.2973)
    [   1][  60] Batch Time (avg: 0.2529, sd: 0.0047) Speed (avg: 1012.0795) Data (avg: 1.0539, sd: 2.1081)
    [   1][  70] Batch Time (avg: 0.2526, sd: 0.0045) Speed (avg: 1013.4748) Data (avg: 0.9749, sd: 1.9661)
    [   1][  80] Batch Time (avg: 0.2529, sd: 0.0055) Speed (avg: 1012.3729) Data (avg: 0.8931, sd: 1.8561)
    [   1][  90] Batch Time (avg: 0.2528, sd: 0.0054) Speed (avg: 1012.7439) Data (avg: 0.8536, sd: 1.7675)
    [   1][ 100] Batch Time (avg: 0.2528, sd: 0.0057) Speed (avg: 1012.7326) Data (avg: 0.7990, sd: 1.6911)
                      Metric ,    Average , Deviation ,        Min ,        Max , count ,  half , batch , workers , loader ,    model , hostname ,                  GPU 
        Waiting for data (s) ,     1.2542 ,    2.1999 ,     0.0001 ,     7.7139 ,    99 , False ,   256 ,       4 ,  torch , resnet18 ,  kepler5 , Tesla V100-PCIE-16GB 
        GPU Compute Time (s) ,     0.2539 ,    0.0084 ,     0.2488 ,     0.3026 ,    99 , False ,   256 ,       4 ,  torch , resnet18 ,  kepler5 , Tesla V100-PCIE-16GB 
         Full Batch Time (s) ,     1.5098 ,    2.2044 ,     0.2492 ,     7.9706 ,    99 , False ,   256 ,       4 ,  torch , resnet18 ,  kepler5 , Tesla V100-PCIE-16GB 
       Compute Speed (img/s) ,  1004.3252 ,        NA ,   845.0368 ,  1028.4588 ,    99 , False ,   256 ,       4 ,  torch , resnet18 ,  kepler5 , Tesla V100-PCIE-16GB 
     Effective Speed (img/s) ,   169.5537 ,        NA ,    32.1179 ,  1027.2898 ,    99 , False ,   256 ,       4 ,  torch , resnet18 ,  kepler5 , Tesla V100-PCIE-16GB 
         gpu.temperature.gpu ,    44.4568 ,    2.1591 ,    41.0000 ,    50.0000 ,   440 , False ,   256 ,       4 ,  torch , resnet18 ,  kepler5 , Tesla V100-PCIE-16GB 
         gpu.utilization.gpu ,    23.0386 ,   39.5398 ,     0.0000 ,    99.0000 ,   440 , False ,   256 ,       4 ,  torch , resnet18 ,  kepler5 , Tesla V100-PCIE-16GB 
      gpu.utilization.memory ,    11.8318 ,   20.7916 ,     0.0000 ,    59.0000 ,   440 , False ,   256 ,       4 ,  torch , resnet18 ,  kepler5 , Tesla V100-PCIE-16GB 
            gpu.memory.total , 16160.0000 ,    0.0000 , 16160.0000 , 16160.0000 ,   440 , False ,   256 ,       4 ,  torch , resnet18 ,  kepler5 , Tesla V100-PCIE-16GB 
             gpu.memory.free ,  6249.6818 , 1436.2063 ,  5981.0000 , 15131.0000 ,   440 , False ,   256 ,       4 ,  torch , resnet18 ,  kepler5 , Tesla V100-PCIE-16GB 
             gpu.memory.used ,  9910.3182 , 1436.2063 ,  1029.0000 , 10179.0000 ,   440 , False ,   256 ,       4 ,  torch , resnet18 ,  kepler5 , Tesla V100-PCIE-16GB 
