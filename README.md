Mixed Precision
===============

# PyTorch

## Expectation

The expected speed up gain should be around 2x according to [this][1]

[1]: https://devblogs.nvidia.com/mixed-precision-resnet-50-tensor-cores/


## Benchmark - Apex Examples

### Process

    > nvidia-docker run --ipc=host -it -v /Tmp/delaunap:/data --rm nvcr.io/nvidia/pytorch:18.07-py3
    > git clone https://github.com/NVIDIA/apex.git
    > cd apex/examples/imagenet
    
    > python main.py -a resnet18 -j 4 -b 128 --static-loss-scale 256 /data/img_net/ 
    > python main.py -a resnet18 -j 4 -b 128 --static-loss-scale 256 /data/img_net/ --fp16
    > python main.py -a resnet50 -j 4 -b 128 --static-loss-scale 256 /data/img_net/
    > python main.py -a resnet50 -j 4 -b 128 --static-loss-scale 256 /data/img_net/ --fp16
    

* Slurm: pytorch 0.4.1
* Container: pytorch 0.5a
* GPU: Tesla V100-PCIE-16GB

### Results 

* Average Speed after 360 batch of 128 images, higher is better

      
|   Speed   |  Slurm  | 16/32 | Container | 16/32 | Container/Slurm |
|----------:|--------:|------:|----------:|------:|----------------:|  
|Resnet18-32|    192  |       |       771 |       |            4.02 |
|Resnet18-16|    194  |  1.01 |       805 |  1.04 |            4.15 |
|Resnet50-32|    175  |       |       333 |       |            1.90 |
|Resnet50-16|    181  |  1.03 |       672 |  2.02 |            3.71 |

* Resnet-18 speed seems limited by data loading. 


## Benchmark

### Process

    > nvidia-docker run --ipc=host -it -v /Tmp/delaunap:/data --rm nvcr.io/nvidia/pytorch:18.07-py3
    > git clone https://github.com/Delaunay/MixedPrecision
    > cd MixedPrecision
    > python setup.py install 
    > resnet-18-pt -j 4 -b 128 --static-loss-scale 128 --data /data/img_net/ --gpu --prof 10
    > resnet-18-pt -j 4 -b 128 --static-loss-scale 128 --data /data/img_net/ --gpu --prof 10 --half
    > resnet-50-pt -j 4 -b 128 --static-loss-scale 128 --data /data/img_net/ --gpu --prof 10
    > resnet-50-pt -j 4 -b 128 --static-loss-scale 128 --data /data/img_net/ --gpu --prof 10 --half
    
 ### Results

|   Speed   |    B=128 | 16/32 |     B=256 | 16/32 | 
|----------:|---------:|------:|----------:|------:|
|Resnet18-32| 1267.474 |       | 1430.7821 |       |
|Resnet18-16| 2991.365 |  2.36 | 3105.9595 |  2.17 | 
|Resnet50-32|  345.037 |       |           |       |
|Resnet50-16|  732.320 |  2.12 |  805.9883 |       |

* Resnet18 - 128 - 32: `min: 1080 img/s`  `max: 2922 img/s`
* Resnet18 - 128 - 16: `min: 1899 img/s`  `max: 4812 img/s`
* Resnet18 - 256 - 32: `min: 1093 img/s`  `max: 3368 img/s`
* Resnet18 - 256 - 16: `min: 2269 img/s`  `max: 5871 img/s`

## Conclusion

Pytorch next version should be significantly faster and will improve mixed precision support.
It is unclear why I am able to get greater performances on my end. I think the data loading is faster for me.
On both side Resnet18 shows a lot of variance in term of speed, I think this is caused by data loading waiting time.


## Procedure

* use a Pytorch dev version (0.4.1 will not do) or a NVIDIA container
* install Apex by NVIDIA
* Convert your model to half using the function `network_to_half`
  * `model = apex.fp16_utils.network_to_half(model)`
  DO NOT use `.half()` not every layer should be converted to half precision
* Enable benchmark mode if your model is static
  * `torch.backends.cudnn.benchmark = True`
* use `apex.fp16_utils.fp16_optimizer.FP16_Optimizer` to wrap your optimizer
  * `FP16_Optimizer(SGD(...), static_loss_scale=256)`
  
Note:

* Mixed precision reduce the memory footprint of your model; increase batch size!
* Half precision has a very small precision. Make sure the scaling is appropriate or your model will not train
* Training your model using half precision is not equivalent to single precision. Accuracy will be different



# Tensorflow - TODO

## Expectation

The expected speed up gain should be around 2x according to [this][1]

[1]: https://devblogs.nvidia.com/mixed-precision-resnet-50-tensor-cores/

## Process

    nvidia-docker run --ipc=host -it -v /Tmp/delaunap:/data --rm nvcr.io/nvidia/tensorflow:18.07-py3

    

   
## Results


|           |  Slurm  | 16/32 | Container | 16/3 | Container/Slurm |
|----------:|--------:|------:|----------:|-----:|----------------:|  
|Resnet18-32|      |       |        |      |             |
|Resnet18-16|      |   |        |  |             |
|Resnet50-32|      |       |        |      |             |
|Resnet50-16|      |   |        |  |             |



# DALI Pytorch

## No Dali

### Pytorch 0.4.1
    > resnet-18-pt -j 4 -b 256 --static-loss-scale 256 --data /Tmp/delaunap/img_net/ --gpu --prof 10 --half
    
    [   1][  10] Batch Time (avg: 0.1401, sd: 0.0000) Speed (avg: 1826.7968) Data (avg: 0.8276, sd: 1.1526)
    [   1][  20] Batch Time (avg: 0.0958, sd: 0.0421) Speed (avg: 2672.7961) Data (avg: 0.9024, sd: 1.0729)
    [   1][  30] Batch Time (avg: 0.0983, sd: 0.0412) Speed (avg: 2604.9385) Data (avg: 0.9169, sd: 1.0719)
    [   1][  40] Batch Time (avg: 0.0995, sd: 0.0406) Speed (avg: 2572.6137) Data (avg: 0.9587, sd: 1.0955)
    [   1][  50] Batch Time (avg: 0.0992, sd: 0.0401) Speed (avg: 2580.0874) Data (avg: 0.9704, sd: 1.1174)
    [   1][  60] Batch Time (avg: 0.1022, sd: 0.0394) Speed (avg: 2503.9901) Data (avg: 0.9756, sd: 1.3095)
    [   1][  70] Batch Time (avg: 0.1005, sd: 0.0404) Speed (avg: 2546.4370) Data (avg: 1.0532, sd: 1.4395)
    [   1][  80] Batch Time (avg: 0.0990, sd: 0.0404) Speed (avg: 2584.6762) Data (avg: 1.1112, sd: 1.4638)
    [   1][  90] Batch Time (avg: 0.0978, sd: 0.0404) Speed (avg: 2617.4232) Data (avg: 1.1394, sd: 1.4736)
    [   1][ 100] Batch Time (avg: 0.0972, sd: 0.0408) Speed (avg: 2633.4100) Data (avg: 1.1299, sd: 1.5109)
    Data Loading (CPU) (avg: 1.2034, sd: 1.5823)
    Data Loading (GPU) (avg: 0.0088, sd: 0.0035)
    [   1] Epoch Time (avg: 134.0267, sd: 0.0000) Batch Time (max: 0.1585, min: 0.0474) Loss: 7.0273
    
### Pytorch 0.5.a
    
    [   1][  10] Batch Time (avg: 0.0821, sd: 0.0000) Speed (avg: 3117.3461) Data (avg: 0.0037, sd: 0.0084)
    [   1][  20] Batch Time (avg: 0.0833, sd: 0.0201) Speed (avg: 3073.8330) Data (avg: 0.0398, sd: 0.0843)
    [   1][  30] Batch Time (avg: 0.0763, sd: 0.0230) Speed (avg: 3357.1784) Data (avg: 0.0749, sd: 0.1169)
    [   1][  40] Batch Time (avg: 0.0799, sd: 0.0240) Speed (avg: 3205.6393) Data (avg: 0.0814, sd: 0.1331)
    [   1][  50] Batch Time (avg: 0.0805, sd: 0.0249) Speed (avg: 3179.9347) Data (avg: 0.0908, sd: 0.1432)
    [   1][  60] Batch Time (avg: 0.0821, sd: 0.0259) Speed (avg: 3119.4643) Data (avg: 0.0876, sd: 0.1398)
    [   1][  70] Batch Time (avg: 0.0797, sd: 0.0262) Speed (avg: 3210.1173) Data (avg: 0.0938, sd: 0.1389)
    [   1][  80] Batch Time (avg: 0.0796, sd: 0.0262) Speed (avg: 3214.6918) Data (avg: 0.0943, sd: 0.1391)
    [   1][  90] Batch Time (avg: 0.0786, sd: 0.0260) Speed (avg: 3255.3279) Data (avg: 0.0973, sd: 0.1366)
    [   1][ 100] Batch Time (avg: 0.0785, sd: 0.0258) Speed (avg: 3259.1418) Data (avg: 0.0976, sd: 0.1357)
    Data Loading (CPU) (avg: 0.1061, sd: 0.1574)
    Data Loading (GPU) (avg: 0.0071, sd: 0.0030)
    [   1] Epoch Time (avg: 29.1293, sd: 0.0000) Batch Time (max: 0.1168, min: 0.0478) Loss: 7.0508
        
## With Dali

### Pytorch 0.4.1

    > resnet-18-pt -j 4 -b 256 --static-loss-scale 256 --data /Tmp/delaunap/img_net/ --gpu --prof 10 --half --use-dali

    [   1][  10] Batch Time (avg: 0.1033, sd: 0.0000) Speed (avg: 2478.6799) Data (avg: 0.2429, sd: 0.1116)
    [   1][  20] Batch Time (avg: 0.0974, sd: 0.0196) Speed (avg: 2627.6473) Data (avg: 0.2176, sd: 0.0914)
    [   1][  30] Batch Time (avg: 0.0796, sd: 0.0263) Speed (avg: 3214.3628) Data (avg: 0.3788, sd: 0.4010)
    [   1][  40] Batch Time (avg: 0.0690, sd: 0.0262) Speed (avg: 3710.3386) Data (avg: 0.6082, sd: 0.5300)
    [   1][  50] Batch Time (avg: 0.0637, sd: 0.0245) Speed (avg: 4019.6122) Data (avg: 0.7659, sd: 0.5729)
    [   1][  60] Batch Time (avg: 0.0605, sd: 0.0228) Speed (avg: 4231.7237) Data (avg: 0.8785, sd: 0.5833)
    [   1][  70] Batch Time (avg: 0.0584, sd: 0.0214) Speed (avg: 4385.9566) Data (avg: 0.9559, sd: 0.5756)
    [   1][  80] Batch Time (avg: 0.0568, sd: 0.0201) Speed (avg: 4503.7674) Data (avg: 1.0164, sd: 0.5616)
    [   1][  90] Batch Time (avg: 0.0557, sd: 0.0191) Speed (avg: 4596.1778) Data (avg: 1.0561, sd: 0.5444)
    [   1][ 100] Batch Time (avg: 0.0548, sd: 0.0182) Speed (avg: 4670.6036) Data (avg: 1.0942, sd: 0.5317)
    Data Loading (CPU) (avg: 1.0773, sd: 0.5465)
    Data Loading (GPU) (avg: 0.0102, sd: 0.0015)
    [   1] Epoch Time (avg: 117.1640, sd: 0.0000) Batch Time (max: 0.1305, min: 0.0463) Loss: 6.5391

### Pytorch 0.5.a
    
    [   1][  10] Batch Time (avg: 0.0561, sd: 0.0000) Speed (avg: 4561.2920) Data (avg: 0.1859, sd: 0.0694)
    [   1][  20] Batch Time (avg: 0.0658, sd: 0.0224) Speed (avg: 3892.8965) Data (avg: 0.1792, sd: 0.0565)
    [   1][  30] Batch Time (avg: 0.0618, sd: 0.0165) Speed (avg: 4142.8749) Data (avg: 0.1782, sd: 0.0508)
    [   1][  40] Batch Time (avg: 0.0591, sd: 0.0141) Speed (avg: 4332.1114) Data (avg: 0.1792, sd: 0.0468)
    [   1][  50] Batch Time (avg: 0.0584, sd: 0.0123) Speed (avg: 4380.7632) Data (avg: 0.1821, sd: 0.0459)
    [   1][  60] Batch Time (avg: 0.0600, sd: 0.0171) Speed (avg: 4267.9413) Data (avg: 0.1856, sd: 0.0458)
    [   1][  70] Batch Time (avg: 0.0594, sd: 0.0157) Speed (avg: 4312.8553) Data (avg: 0.1874, sd: 0.0497)
    [   1][  80] Batch Time (avg: 0.0588, sd: 0.0146) Speed (avg: 4350.6058) Data (avg: 0.1866, sd: 0.0471)
    [   1][  90] Batch Time (avg: 0.0584, sd: 0.0138) Speed (avg: 4385.7049) Data (avg: 0.1842, sd: 0.0462)
    [   1][ 100] Batch Time (avg: 0.0578, sd: 0.0131) Speed (avg: 4425.9867) Data (avg: 0.1842, sd: 0.0447)
    Data Loading (CPU) (avg: 0.1804, sd: 0.0514)
    Data Loading (GPU) (avg: 0.0037, sd: 0.0007)
    [   1] Epoch Time (avg: 26.5653, sd: 0.0000) Batch Time (max: 0.1511, min: 0.0507) Loss: 6.5508



