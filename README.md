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

