Mixed Precision
===============

# PyTorch

## Expectation

The expected speed up gain should be around 2x according to [this][1]

[1]: https://devblogs.nvidia.com/mixed-precision-resnet-50-tensor-cores/


## Process

    nvidia-docker run --ipc=host -it -v /Tmp/delaunap:/data --rm nvcr.io/nvidia/pytorch:18.07-py3
    git clone https://github.com/NVIDIA/apex.git
    cd apex/examples/imagenet
    
    python main.py -a resnet18 -j 4 -b 128 --static-loss-scale 256 /data/img_net/ 
    python main.py -a resnet18 -j 4 -b 128 --static-loss-scale 256 /data/img_net/ --fp16
    python main.py -a resnet50 -j 4 -b 128 --static-loss-scale 256 /data/img_net/
    python main.py -a resnet50 -j 4 -b 128 --static-loss-scale 256 /data/img_net/ --fp16
    

* Slurm: pytorch 0.4.1
* Container: pytorch 0.5a


## Results

* Average Speed after 360 batch of 128 images, higher is better

      
|   Speed   |  Slurm  | 16/32 | Container | 16/3 | Container/Slurm |
|----------:|--------:|------:|----------:|-----:|----------------:|  
|Resnet18-32|    192  |       |       771 |      |            4.02 |
|Resnet18-16|    194  |  1.01 |       805 | 1.04 |            4.15 |
|Resnet50-32|    175  |       |       333 |      |            1.90 |
|Resnet50-16|    181  |  1.03 |       672 | 2.02 |            3.71 |


NB: The speed during the training had a lot of variance; which means we might be able to get better speedup in reality 
if we actually do a few epochs. I only did 360 batches.

* Container - Resnet 18-16 `min: 207.920` `max: 2268.350`
* Container - Resnet 18-32 `min: 352.708` `max: 1144.168`
* Slurm - Resnet 18-16 `min: 49.133` `max: 1779.651`
* Slurm - Resnet 18-32 `min: 62.200` `max: 1166.360`
* Slurm - Resnet 50-16 `min: 49.133` `max: 689.778` 
* Slurm - Resnet 50-32 `min: 72.592` `max: 351.653`

The training on the older version of pytorch also showed a lot of variance.


|   Loss    |           Slurm  |       Container |
|----------:|-----------------:|----------------:|  
|Resnet18-32|  4.8026 (5.1898) | 4.8808 (5.1750) |
|Resnet18-16|  5.3828 (6.2438) | 5.4102 (6.2232) |
|Resnet50-32|  5.0502 (5.4794) | 4.9958 (5.4130) |
|Resnet50-16|  5.2891 (5.8204) | 5.3633 (5.7963) |

* unfortunatly the seed was not fixed


## Conclusion

Pytorch next version should be significantly faster and will improve mixed precision support.
Mixed precision only works for big enough networks (conv channel >> 512).


# Tensorflow

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

