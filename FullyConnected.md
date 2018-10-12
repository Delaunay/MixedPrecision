
#PyTorch
## CPU

    root@19ea79fd8130:/workspace/MixedPrecision# mnist-full-pt --shape 3 256 256 --fake
                              half: False
                dynamic_loss_scale: False
                      weight_decay: 0.0001
                        batch_size: 256
                       hidden_size: 64
                           workers: 4
                           permute: False
                        print_freq: 10
                             shape: [3, 256, 256]
                          conv_num: 32
                            epochs: 10
                 static_loss_scale: 1
                              prof: None
                              fake: True
                       kernel_size: 3
                               gpu: False
                        hidden_num: 1
                          momentum: 0.9
                                lr: 0.1
                              data: None
                         GPU Count: 2
                          GPU Name: Tesla V100-PCIE-16GB
    [   1] Compute Time (avg: 14.4253, sd: 0.0000) Loss: 2.3166
    [   2] Compute Time (avg: 13.6492, sd: 0.0000) Loss: 2.3361
    [   3] Compute Time (avg: 13.8762, sd: 0.2270) Loss: 2.3791
    [   4] Compute Time (avg: 14.0496, sd: 0.3074) Loss: 2.3244
    [   5] Compute Time (avg: 14.1234, sd: 0.2953) Loss: 2.3361
    [   6] Compute Time (avg: 14.2392, sd: 0.3512) Loss: 2.3361
    [   7] Compute Time (avg: 14.1500, sd: 0.3776) Loss: 2.3596
    [   8] Compute Time (avg: 14.1418, sd: 0.3502) Loss: 2.3479
    [   9] Compute Time (avg: 14.0626, sd: 0.3888) Loss: 2.3674
    [  10] Compute Time (avg: 14.0554, sd: 0.3671) Loss: 2.3401
   
## GPU
 
    root@19ea79fd8130:/workspace/MixedPrecision# mnist-full-pt --shape 3 256 256 --fake --gpu
                              data: None
                        hidden_num: 1
                           workers: 4
                 static_loss_scale: 1
                          conv_num: 32
                        batch_size: 256
                          momentum: 0.9
                               gpu: True
                       kernel_size: 3
                           permute: False
                              half: False
                              prof: None
                             shape: [3, 256, 256]
                            epochs: 10
                              fake: True
                       hidden_size: 64
                        print_freq: 10
                      weight_decay: 0.0001
                dynamic_loss_scale: False
                                lr: 0.1
                         GPU Count: 2
                          GPU Name: Tesla V100-PCIE-16GB
    [   1] Compute Time (avg: 11.7181, sd: 0.0000) Loss: 2.3166
    [   2] Compute Time (avg: 11.6082, sd: 0.0000) Loss: 2.3362
    [   3] Compute Time (avg: 11.6144, sd: 0.0062) Loss: 2.3791
    [   4] Compute Time (avg: 11.6658, sd: 0.0728) Loss: 2.3244
    [   5] Compute Time (avg: 11.7076, sd: 0.0960) Loss: 2.3362
    [   6] Compute Time (avg: 11.7284, sd: 0.0954) Loss: 2.3362
    [   7] Compute Time (avg: 11.7500, sd: 0.0996) Loss: 2.3596
    [   8] Compute Time (avg: 11.7718, sd: 0.1066) Loss: 2.3479
    [   9] Compute Time (avg: 11.8141, sd: 0.1499) Loss: 2.3674
    [  10] Compute Time (avg: 11.8090, sd: 0.1421) Loss: 2.3401
    
## GPU Half

    root@19ea79fd8130:/workspace/MixedPrecision# mnist-full-pt --shape 3 256 256 --fake --gpu --half
                               gpu: True
                           permute: False
                           workers: 4
                        print_freq: 10
                             shape: [3, 256, 256]
                                lr: 0.1
                          momentum: 0.9
                 static_loss_scale: 1
                              prof: None
                              data: None
                              half: True
                              fake: True
                        batch_size: 256
                dynamic_loss_scale: False
                            epochs: 10
                          conv_num: 32
                        hidden_num: 1
                       hidden_size: 64
                       kernel_size: 3
                      weight_decay: 0.0001
                         GPU Count: 2
                          GPU Name: Tesla V100-PCIE-16GB
    [   1] Compute Time (avg: 11.6878, sd: 0.0000) Loss: 2.3164
    [   2] Compute Time (avg: 10.5726, sd: 0.0000) Loss: 2.3359
    [   3] Compute Time (avg: 10.5364, sd: 0.0362) Loss: 2.3789
    [   4] Compute Time (avg: 10.5322, sd: 0.0301) Loss: 2.3242
    [   5] Compute Time (avg: 10.5285, sd: 0.0269) Loss: 2.3359
    [   6] Compute Time (avg: 10.5270, sd: 0.0243) Loss: 2.3359
    [   7] Compute Time (avg: 10.5230, sd: 0.0238) Loss: 2.3594
    [   8] Compute Time (avg: 10.5214, sd: 0.0224) Loss: 2.3477                                                                                                                                                       
    [   9] Compute Time (avg: 10.5299, sd: 0.0307) Loss: 2.3672                                                                                                                                                       
    [  10] Compute Time (avg: 10.6211, sd: 0.2596) Loss: 2.3398
    
# Tensorflow

## CPU

    root@19ea79fd8130:/workspace/MixedPrecision# mnist-full-tf --shape 3 256 256
                           workers: 4                                                                                                                                                                                 
                          momentum: 0.9                                                                                                                                                                               
                              fake: False                                                                                                                                                                             
                                lr: 0.1                                                                                                                                                                               
                              data: None
                               gpu: False
                        hidden_num: 1
                        print_freq: 10
                      weight_decay: 0.0001
                       kernel_size: 3
                              half: False
                            epochs: 10
                       hidden_size: 64
                             shape: [3, 256, 256]
                 static_loss_scale: 1
                        batch_size: 256
                dynamic_loss_scale: False
                           permute: False
                              prof: None
                          conv_num: 32
    2018-10-11 19:19:13.906871: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1356] Found device 0 with properties: 
    name: Tesla V100-PCIE-16GB major: 7 minor: 0 memoryClockRate(GHz): 1.38
    pciBusID: 0000:03:00.0
    totalMemory: 15.78GiB freeMemory: 15.37GiB
    2018-10-11 19:19:14.073619: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1356] Found device 1 with properties: 
    name: Tesla M40 major: 5 minor: 2 memoryClockRate(GHz): 1.112
    pciBusID: 0000:82:00.0
    totalMemory: 11.18GiB freeMemory: 11.06GiB
    2018-10-11 19:19:14.073681: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1435] Adding visible gpu devices: 0, 1
    2018-10-11 19:19:14.690490: I tensorflow/core/common_runtime/gpu/gpu_device.cc:923] Device interconnect StreamExecutor with strength 1 edge matrix:
    2018-10-11 19:19:14.690555: I tensorflow/core/common_runtime/gpu/gpu_device.cc:929]      0 1 
    2018-10-11 19:19:14.690564: I tensorflow/core/common_runtime/gpu/gpu_device.cc:942] 0:   N N 
    2018-10-11 19:19:14.690569: I tensorflow/core/common_runtime/gpu/gpu_device.cc:942] 1:   N N 
    2018-10-11 19:19:14.691151: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1053] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 14883 MB memory) -> physical GPU (device: 0, name: Tesla V100-PCIE-16GB, pci bus id: 0000:03:00.0, compute capability: 7.0)
    2018-10-11 19:19:14.929602: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1053] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:1 with 10710 MB memory) -> physical GPU (device: 1, name: Tesla M40, pci bus id: 0000:82:00.0, compute capability: 5.2)
    Device mapping:
    /job:localhost/replica:0/task:0/device:XLA_CPU:0 -> device: XLA_CPU device
    /job:localhost/replica:0/task:0/device:XLA_GPU:0 -> device: XLA_GPU device
    /job:localhost/replica:0/task:0/device:GPU:0 -> device: 0, name: Tesla V100-PCIE-16GB, pci bus id: 0000:03:00.0, compute capability: 7.0
    /job:localhost/replica:0/task:0/device:GPU:1 -> device: 1, name: Tesla M40, pci bus id: 0000:82:00.0, compute capability: 5.2
    2018-10-11 19:19:15.208642: I tensorflow/core/common_runtime/direct_session.cc:284] Device mapping:
    /job:localhost/replica:0/task:0/device:XLA_CPU:0 -> device: XLA_CPU device
    /job:localhost/replica:0/task:0/device:XLA_GPU:0 -> device: XLA_GPU device
    /job:localhost/replica:0/task:0/device:GPU:0 -> device: 0, name: Tesla V100-PCIE-16GB, pci bus id: 0000:03:00.0, compute capability: 7.0
    /job:localhost/replica:0/task:0/device:GPU:1 -> device: 1, name: Tesla M40, pci bus id: 0000:82:00.0, compute capability: 5.2
    
    fp32_storage/fp32_storage/obiases/Momentum: (VariableV2): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.216307: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/fp32_storage/obiases/Momentum: (VariableV2)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/fp32_storage/obiases/Momentum/read: (Identity): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.216350: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/fp32_storage/obiases/Momentum/read: (Identity)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/fp32_storage/obiases/Momentum/Assign: (Assign): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.216373: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/fp32_storage/obiases/Momentum/Assign: (Assign)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/fp32_storage/oweights/Momentum: (VariableV2): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.216397: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/fp32_storage/oweights/Momentum: (VariableV2)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/fp32_storage/oweights/Momentum/read: (Identity): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.216424: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/fp32_storage/oweights/Momentum/read: (Identity)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/fp32_storage/oweights/Momentum/Assign: (Assign): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.216452: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/fp32_storage/oweights/Momentum/Assign: (Assign)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/fp32_storage/ibiases/Momentum: (VariableV2): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.216482: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/fp32_storage/ibiases/Momentum: (VariableV2)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/fp32_storage/ibiases/Momentum/read: (Identity): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.216515: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/fp32_storage/ibiases/Momentum/read: (Identity)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/fp32_storage/ibiases/Momentum/Assign: (Assign): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.216547: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/fp32_storage/ibiases/Momentum/Assign: (Assign)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/fp32_storage/iweights/Momentum: (VariableV2): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.216595: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/fp32_storage/iweights/Momentum: (VariableV2)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/fp32_storage/iweights/Momentum/read: (Identity): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.216629: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/fp32_storage/iweights/Momentum/read: (Identity)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/fp32_storage/iweights/Momentum/Initializer/zeros: (Fill): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.216665: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/fp32_storage/iweights/Momentum/Initializer/zeros: (Fill)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/fp32_storage/iweights/Momentum/Assign: (Assign): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.216701: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/fp32_storage/iweights/Momentum/Assign: (Assign)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/gradients/fp32_storage/input_layer/add_grad/BroadcastGradientArgs: (BroadcastGradientArgs): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.216737: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/input_layer/add_grad/BroadcastGradientArgs: (BroadcastGradientArgs)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/gradients/fp32_storage/output_layer/add_grad/BroadcastGradientArgs: (BroadcastGradientArgs): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.216769: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/output_layer/add_grad/BroadcastGradientArgs: (BroadcastGradientArgs)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Mul_grad/BroadcastGradientArgs: (BroadcastGradientArgs): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.216805: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Mul_grad/BroadcastGradientArgs: (BroadcastGradientArgs)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/div_grad/BroadcastGradientArgs: (BroadcastGradientArgs): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.216839: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/div_grad/BroadcastGradientArgs: (BroadcastGradientArgs)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/gradients/Fill: (Fill): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.216874: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/Fill: (Fill)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/gradients/fp32_storage/mul_grad/Mul: (Mul): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.216909: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/mul_grad/Mul: (Mul)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success: (NoOp): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.216942: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success: (NoOp)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/softmax_cross_entropy_loss/ones_like: (Fill): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.216973: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/ones_like: (Fill)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success: (NoOp): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.217008: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success: (NoOp)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/softmax_cross_entropy_loss/num_present/broadcast_weights/ones_like: (Fill): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.217038: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/num_present/broadcast_weights/ones_like: (Fill)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/softmax_cross_entropy_loss/num_present/ones_like: (Fill): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.217061: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/num_present/ones_like: (Fill)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/softmax_cross_entropy_loss/num_present/Equal: (Equal): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.217089: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/num_present/Equal: (Equal)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/softmax_cross_entropy_loss/num_present/Select: (Select): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.217124: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/num_present/Select: (Select)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/softmax_cross_entropy_loss/num_present/broadcast_weights: (Mul): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.217157: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/num_present/broadcast_weights: (Mul)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/softmax_cross_entropy_loss/num_present: (Sum): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.217191: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/num_present: (Sum)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/softmax_cross_entropy_loss/Equal: (Equal): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.217221: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/Equal: (Equal)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/softmax_cross_entropy_loss/Select: (Select): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.217251: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/Select: (Select)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/softmax_cross_entropy_loss/Greater: (Greater): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.217278: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/Greater: (Greater)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/value_grad/Select_1: (Select): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.217303: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/value_grad/Select_1: (Select)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/value_grad/Select: (Select): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.217333: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/value_grad/Select: (Select)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/div_grad/RealDiv: (RealDiv): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.217368: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/div_grad/RealDiv: (RealDiv)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/div_grad/Sum: (Sum): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.217395: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/div_grad/Sum: (Sum)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/div_grad/Reshape: (Reshape): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.217429: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/div_grad/Reshape: (Reshape)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Sum_1_grad/Reshape: (Reshape): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.217468: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Sum_1_grad/Reshape: (Reshape)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Sum_1_grad/Tile: (Tile): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.217508: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Sum_1_grad/Tile: (Tile)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Sum_grad/Reshape: (Reshape): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.217539: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Sum_grad/Reshape: (Reshape)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Sum_grad/Tile: (Tile): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.217573: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Sum_grad/Tile: (Tile)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Mul_grad/Mul: (Mul): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.217604: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Mul_grad/Mul: (Mul)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Mul_grad/Sum: (Sum): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.217633: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Mul_grad/Sum: (Sum)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Mul_grad/Reshape: (Reshape): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.217657: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Mul_grad/Reshape: (Reshape)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/xentropy/Reshape_2_grad/Reshape: (Reshape): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.217684: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/xentropy/Reshape_2_grad/Reshape: (Reshape)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/xentropy_grad/ExpandDims_1: (ExpandDims): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.217717: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/xentropy_grad/ExpandDims_1: (ExpandDims)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/xentropy_grad/ExpandDims: (ExpandDims): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.217754: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/xentropy_grad/ExpandDims: (ExpandDims)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/softmax_cross_entropy_loss/xentropy/Sub_1: (Sub): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.217791: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/xentropy/Sub_1: (Sub)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/softmax_cross_entropy_loss/xentropy/Slice_1/begin: (Pack): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.217830: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/xentropy/Slice_1/begin: (Pack)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/softmax_cross_entropy_loss/xentropy/Slice_1: (Slice): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.217867: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/xentropy/Slice_1: (Slice)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/softmax_cross_entropy_loss/xentropy/concat_1: (ConcatV2): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.217902: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/xentropy/concat_1: (ConcatV2)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/softmax_cross_entropy_loss/xentropy/Sub: (Sub): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.217954: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/xentropy/Sub: (Sub)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/softmax_cross_entropy_loss/xentropy/Slice/begin: (Pack): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.217991: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/xentropy/Slice/begin: (Pack)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/softmax_cross_entropy_loss/xentropy/Slice: (Slice): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.218026: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/xentropy/Slice: (Slice)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/softmax_cross_entropy_loss/xentropy/concat: (ConcatV2): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.218066: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/xentropy/concat: (ConcatV2)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/softmax_cross_entropy_loss/xentropy/Sub_2: (Sub): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.218099: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/xentropy/Sub_2: (Sub)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/softmax_cross_entropy_loss/xentropy/Slice_2/size: (Pack): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.218121: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/xentropy/Slice_2/size: (Pack)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/softmax_cross_entropy_loss/xentropy/Slice_2: (Slice): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.218151: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/xentropy/Slice_2: (Slice)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/softmax_cross_entropy_loss/labels_stop_gradient: (StopGradient): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.218185: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/labels_stop_gradient: (StopGradient)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/softmax_cross_entropy_loss/xentropy/Reshape_1: (Reshape): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.218216: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/xentropy/Reshape_1: (Reshape)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/obiases: (VariableV2): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.218242: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/obiases: (VariableV2)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/obiases/read: (Identity): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.218271: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/obiases/read: (Identity)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/obiases/Assign: (Assign): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.218301: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/obiases/Assign: (Assign)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/oweights: (VariableV2): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.218348: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/oweights: (VariableV2)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/oweights/read: (Identity): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.218389: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/oweights/read: (Identity)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/oweights/Initializer/random_uniform/sub: (Sub): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.218423: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/oweights/Initializer/random_uniform/sub: (Sub)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/oweights/Initializer/random_uniform/RandomUniform: (RandomUniform): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.218456: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/oweights/Initializer/random_uniform/RandomUniform: (RandomUniform)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/oweights/Initializer/random_uniform/mul: (Mul): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.218491: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/oweights/Initializer/random_uniform/mul: (Mul)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/oweights/Initializer/random_uniform: (Add): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.218527: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/oweights/Initializer/random_uniform: (Add)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/oweights/Assign: (Assign): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.218557: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/oweights/Assign: (Assign)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/ibiases: (VariableV2): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.218587: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/ibiases: (VariableV2)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/ibiases/read: (Identity): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.218617: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/ibiases/read: (Identity)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/ibiases/Assign: (Assign): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.218643: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/ibiases/Assign: (Assign)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/iweights: (VariableV2): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.218675: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/iweights: (VariableV2)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/iweights/read: (Identity): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.218702: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/iweights/read: (Identity)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/iweights/Initializer/random_uniform/sub: (Sub): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.218732: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/iweights/Initializer/random_uniform/sub: (Sub)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/iweights/Initializer/random_uniform/RandomUniform: (RandomUniform): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.218764: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/iweights/Initializer/random_uniform/RandomUniform: (RandomUniform)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/iweights/Initializer/random_uniform/mul: (Mul): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.218796: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/iweights/Initializer/random_uniform/mul: (Mul)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/iweights/Initializer/random_uniform: (Add): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.218828: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/iweights/Initializer/random_uniform: (Add)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/iweights/Assign: (Assign): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.218875: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/iweights/Assign: (Assign)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/init: (NoOp): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.218908: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/init: (NoOp)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/input_layer/MatMul: (MatMul): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.218936: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/input_layer/MatMul: (MatMul)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/input_layer/add: (Add): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.218962: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/input_layer/add: (Add)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/input_layer/Relu: (Relu): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.218989: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/input_layer/Relu: (Relu)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/output_layer/MatMul: (MatMul): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.219017: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/output_layer/MatMul: (MatMul)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/output_layer/add: (Add): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.219043: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/output_layer/add: (Add)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/softmax_cross_entropy_loss/xentropy/Reshape: (Reshape): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.219077: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/xentropy/Reshape: (Reshape)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/xentropy_grad/LogSoftmax: (LogSoftmax): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.219111: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/xentropy_grad/LogSoftmax: (LogSoftmax)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/xentropy_grad/Neg: (Neg): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.219140: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/xentropy_grad/Neg: (Neg)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/xentropy_grad/mul_1: (Mul): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.219170: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/xentropy_grad/mul_1: (Mul)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/softmax_cross_entropy_loss/xentropy: (SoftmaxCrossEntropyWithLogits): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.219203: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/xentropy: (SoftmaxCrossEntropyWithLogits)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/xentropy_grad/mul: (Mul): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.219237: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/xentropy_grad/mul: (Mul)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/xentropy/Reshape_grad/Reshape: (Reshape): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.219262: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/xentropy/Reshape_grad/Reshape: (Reshape)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/gradients/fp32_storage/output_layer/add_grad/Sum_1: (Sum): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.219294: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/output_layer/add_grad/Sum_1: (Sum)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/gradients/fp32_storage/output_layer/add_grad/Reshape_1: (Reshape): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.219328: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/output_layer/add_grad/Reshape_1: (Reshape)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/truediv_3: (RealDiv): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.219372: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/truediv_3: (RealDiv)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/Momentum/update_fp32_storage/obiases/ApplyMomentum: (ApplyMomentum): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.219406: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/Momentum/update_fp32_storage/obiases/ApplyMomentum: (ApplyMomentum)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/gradients/fp32_storage/output_layer/add_grad/Sum: (Sum): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.219441: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/output_layer/add_grad/Sum: (Sum)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/gradients/fp32_storage/output_layer/add_grad/Reshape: (Reshape): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.219475: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/output_layer/add_grad/Reshape: (Reshape)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/gradients/fp32_storage/output_layer/MatMul_grad/MatMul_1: (MatMul): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.219520: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/output_layer/MatMul_grad/MatMul_1: (MatMul)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/truediv_2: (RealDiv): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.219575: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/truediv_2: (RealDiv)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/Momentum/update_fp32_storage/oweights/ApplyMomentum: (ApplyMomentum): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.219609: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/Momentum/update_fp32_storage/oweights/ApplyMomentum: (ApplyMomentum)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/gradients/fp32_storage/output_layer/MatMul_grad/MatMul: (MatMul): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.219635: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/output_layer/MatMul_grad/MatMul: (MatMul)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/gradients/fp32_storage/input_layer/Relu_grad/ReluGrad: (ReluGrad): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.219666: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/input_layer/Relu_grad/ReluGrad: (ReluGrad)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/gradients/fp32_storage/input_layer/add_grad/Sum_1: (Sum): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.219698: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/input_layer/add_grad/Sum_1: (Sum)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/gradients/fp32_storage/input_layer/add_grad/Reshape_1: (Reshape): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.219733: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/input_layer/add_grad/Reshape_1: (Reshape)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/truediv_1: (RealDiv): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.219768: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/truediv_1: (RealDiv)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/Momentum/update_fp32_storage/ibiases/ApplyMomentum: (ApplyMomentum): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.219798: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/Momentum/update_fp32_storage/ibiases/ApplyMomentum: (ApplyMomentum)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/gradients/fp32_storage/input_layer/add_grad/Sum: (Sum): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.219830: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/input_layer/add_grad/Sum: (Sum)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/gradients/fp32_storage/input_layer/add_grad/Reshape: (Reshape): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.219860: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/input_layer/add_grad/Reshape: (Reshape)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/gradients/fp32_storage/input_layer/MatMul_grad/MatMul_1: (MatMul): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.219891: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/input_layer/MatMul_grad/MatMul_1: (MatMul)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/truediv: (RealDiv): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.219953: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/truediv: (RealDiv)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/Momentum/update_fp32_storage/iweights/ApplyMomentum: (ApplyMomentum): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.219984: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/Momentum/update_fp32_storage/iweights/ApplyMomentum: (ApplyMomentum)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/Momentum: (NoOp): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.220012: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/Momentum: (NoOp)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/gradients/fp32_storage/input_layer/MatMul_grad/MatMul: (MatMul): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.220040: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/input_layer/MatMul_grad/MatMul: (MatMul)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/gradients/zeros_like: (ZerosLike): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.220067: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/zeros_like: (ZerosLike)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/softmax_cross_entropy_loss/xentropy/Reshape_2: (Reshape): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.220094: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/xentropy/Reshape_2: (Reshape)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Mul_grad/Mul_1: (Mul): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.220125: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Mul_grad/Mul_1: (Mul)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Mul_grad/Sum_1: (Sum): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.220150: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Mul_grad/Sum_1: (Sum)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Mul_grad/Reshape_1: (Reshape): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.220181: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Mul_grad/Reshape_1: (Reshape)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/softmax_cross_entropy_loss/Mul: (Mul): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.220234: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/Mul: (Mul)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/softmax_cross_entropy_loss/Sum: (Sum): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.220274: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/Sum: (Sum)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/softmax_cross_entropy_loss/Sum_1: (Sum): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.220303: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/Sum_1: (Sum)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/div_grad/Neg: (Neg): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.220336: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/div_grad/Neg: (Neg)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/div_grad/RealDiv_1: (RealDiv): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.220369: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/div_grad/RealDiv_1: (RealDiv)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/div_grad/RealDiv_2: (RealDiv): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.220403: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/div_grad/RealDiv_2: (RealDiv)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/div_grad/mul: (Mul): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.220437: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/div_grad/mul: (Mul)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/div_grad/Sum_1: (Sum): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.220471: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/div_grad/Sum_1: (Sum)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/div_grad/Reshape_1: (Reshape): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.220506: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/div_grad/Reshape_1: (Reshape)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/softmax_cross_entropy_loss/div: (RealDiv): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.220540: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/div: (RealDiv)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/softmax_cross_entropy_loss/value: (Select): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.220572: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/value: (Select)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/gradients/fp32_storage/mul_grad/Mul_1: (Mul): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.220600: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/mul_grad/Mul_1: (Mul)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/mul: (Mul): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.220630: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/mul: (Mul)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/Momentum/momentum: (Const): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.220656: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/Momentum/momentum: (Const)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/Momentum/learning_rate: (Const): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.220681: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/Momentum/learning_rate: (Const)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/fp32_storage/obiases/Momentum/Initializer/zeros: (Const): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.220709: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/fp32_storage/obiases/Momentum/Initializer/zeros: (Const)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/fp32_storage/oweights/Momentum/Initializer/zeros: (Const): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.220732: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/fp32_storage/oweights/Momentum/Initializer/zeros: (Const)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/fp32_storage/ibiases/Momentum/Initializer/zeros: (Const): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.220756: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/fp32_storage/ibiases/Momentum/Initializer/zeros: (Const)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/fp32_storage/iweights/Momentum/Initializer/zeros/Const: (Const): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.220778: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/fp32_storage/iweights/Momentum/Initializer/zeros/Const: (Const)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/fp32_storage/iweights/Momentum/Initializer/zeros/shape_as_tensor: (Const): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.220802: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/fp32_storage/iweights/Momentum/Initializer/zeros/shape_as_tensor: (Const)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/truediv_3/y: (Const): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.220833: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/truediv_3/y: (Const)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/truediv_2/y: (Const): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.220862: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/truediv_2/y: (Const)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/truediv_1/y: (Const): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.220894: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/truediv_1/y: (Const)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/truediv/y: (Const): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.220923: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/truediv/y: (Const)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/gradients/fp32_storage/input_layer/add_grad/Shape_1: (Const): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.220959: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/input_layer/add_grad/Shape_1: (Const)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/gradients/fp32_storage/input_layer/add_grad/Shape: (Const): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.220991: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/input_layer/add_grad/Shape: (Const)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/gradients/fp32_storage/output_layer/add_grad/Shape_1: (Const): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.221018: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/output_layer/add_grad/Shape_1: (Const)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/gradients/fp32_storage/output_layer/add_grad/Shape: (Const): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.221044: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/output_layer/add_grad/Shape: (Const)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/xentropy/Reshape_grad/Shape: (Const): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.221072: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/xentropy/Reshape_grad/Shape: (Const)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/xentropy_grad/ExpandDims_1/dim: (Const): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.221104: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/xentropy_grad/ExpandDims_1/dim: (Const)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/xentropy_grad/ExpandDims/dim: (Const): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.221137: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/xentropy_grad/ExpandDims/dim: (Const)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/xentropy/Reshape_2_grad/Shape: (Const): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.221174: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/xentropy/Reshape_2_grad/Shape: (Const)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Mul_grad/Shape_1: (Const): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.221209: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Mul_grad/Shape_1: (Const)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Mul_grad/Shape: (Const): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.221242: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Mul_grad/Shape: (Const)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Sum_grad/Const: (Const): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.221275: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Sum_grad/Const: (Const)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Sum_grad/Reshape/shape: (Const): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.221311: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Sum_grad/Reshape/shape: (Const)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Sum_1_grad/Const: (Const): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.221345: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Sum_1_grad/Const: (Const)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Sum_1_grad/Reshape/shape: (Const): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.221380: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Sum_1_grad/Reshape/shape: (Const)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/div_grad/Shape_1: (Const): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.221412: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/div_grad/Shape_1: (Const)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/div_grad/Shape: (Const): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.221446: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/div_grad/Shape: (Const)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/value_grad/zeros_like: (Const): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.221479: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/value_grad/zeros_like: (Const)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/gradients/grad_ys_0: (Const): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.221514: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/grad_ys_0: (Const)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/gradients/Shape: (Const): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.221546: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/Shape: (Const)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/mul/y: (Const): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.221579: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/mul/y: (Const)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/softmax_cross_entropy_loss/zeros_like: (Const): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.221613: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/zeros_like: (Const)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/softmax_cross_entropy_loss/ones_like/Const: (Const): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.221645: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/ones_like/Const: (Const)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/softmax_cross_entropy_loss/ones_like/Shape: (Const): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.221673: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/ones_like/Shape: (Const)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/softmax_cross_entropy_loss/Equal/y: (Const): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.221703: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/Equal/y: (Const)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/softmax_cross_entropy_loss/Greater/y: (Const): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.221729: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/Greater/y: (Const)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/softmax_cross_entropy_loss/Const_1: (Const): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.221759: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/Const_1: (Const)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/softmax_cross_entropy_loss/num_present/Const: (Const): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.221794: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/num_present/Const: (Const)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/softmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/Const: (Const): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.221836: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/Const: (Const)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/softmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/Shape: (Const): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.221858: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/Shape: (Const)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/values/rank: (Const): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.221879: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/values/rank: (Const)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/values/shape: (Const): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.221900: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/values/shape: (Const)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/weights/rank: (Const): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.221927: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/weights/rank: (Const)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/weights/shape: (Const): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.221956: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/weights/shape: (Const)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/softmax_cross_entropy_loss/num_present/ones_like/Const: (Const): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.221986: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/num_present/ones_like/Const: (Const)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/softmax_cross_entropy_loss/num_present/ones_like/Shape: (Const): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.222021: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/num_present/ones_like/Shape: (Const)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/softmax_cross_entropy_loss/num_present/zeros_like: (Const): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.222052: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/num_present/zeros_like: (Const)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/softmax_cross_entropy_loss/num_present/Equal/y: (Const): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.222078: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/num_present/Equal/y: (Const)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/softmax_cross_entropy_loss/Const: (Const): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.222098: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/Const: (Const)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/softmax_cross_entropy_loss/ToFloat_1/x: (Const): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.222118: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/ToFloat_1/x: (Const)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/softmax_cross_entropy_loss/assert_broadcastable/values/rank: (Const): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.222137: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/assert_broadcastable/values/rank: (Const)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/softmax_cross_entropy_loss/assert_broadcastable/values/shape: (Const): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.222161: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/assert_broadcastable/values/shape: (Const)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/softmax_cross_entropy_loss/assert_broadcastable/weights/rank: (Const): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.222193: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/assert_broadcastable/weights/rank: (Const)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/softmax_cross_entropy_loss/assert_broadcastable/weights/shape: (Const): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.222228: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/assert_broadcastable/weights/shape: (Const)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/softmax_cross_entropy_loss/assert_broadcastable/weights: (Const): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.222261: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/assert_broadcastable/weights: (Const)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/softmax_cross_entropy_loss/xentropy/Slice_2/begin: (Const): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.222295: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/xentropy/Slice_2/begin: (Const)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/softmax_cross_entropy_loss/xentropy/Sub_2/y: (Const): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.222328: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/xentropy/Sub_2/y: (Const)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/softmax_cross_entropy_loss/xentropy/concat_1/axis: (Const): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.222380: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/xentropy/concat_1/axis: (Const)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/softmax_cross_entropy_loss/xentropy/concat_1/values_0: (Const): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.222415: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/xentropy/concat_1/values_0: (Const)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/softmax_cross_entropy_loss/xentropy/Slice_1/size: (Const): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.222446: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/xentropy/Slice_1/size: (Const)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/softmax_cross_entropy_loss/xentropy/Sub_1/y: (Const): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.222476: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/xentropy/Sub_1/y: (Const)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/softmax_cross_entropy_loss/xentropy/Shape_2: (Const): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.222505: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/xentropy/Shape_2: (Const)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/softmax_cross_entropy_loss/xentropy/Rank_2: (Const): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.222536: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/xentropy/Rank_2: (Const)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/softmax_cross_entropy_loss/xentropy/concat/axis: (Const): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.222564: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/xentropy/concat/axis: (Const)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/softmax_cross_entropy_loss/xentropy/concat/values_0: (Const): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.222591: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/xentropy/concat/values_0: (Const)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/softmax_cross_entropy_loss/xentropy/Slice/size: (Const): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.222614: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/xentropy/Slice/size: (Const)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/softmax_cross_entropy_loss/xentropy/Sub/y: (Const): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.222633: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/xentropy/Sub/y: (Const)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/softmax_cross_entropy_loss/xentropy/Shape_1: (Const): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.222663: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/xentropy/Shape_1: (Const)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/softmax_cross_entropy_loss/xentropy/Rank_1: (Const): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.222693: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/xentropy/Rank_1: (Const)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/softmax_cross_entropy_loss/xentropy/Shape: (Const): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.222725: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/xentropy/Shape: (Const)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/softmax_cross_entropy_loss/xentropy/Rank: (Const): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.222760: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/xentropy/Rank: (Const)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/Placeholder_1: (Placeholder): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.222792: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/Placeholder_1: (Placeholder)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/obiases/Initializer/zeros: (Const): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.222822: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/obiases/Initializer/zeros: (Const)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/oweights/Initializer/random_uniform/max: (Const): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.222850: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/oweights/Initializer/random_uniform/max: (Const)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/oweights/Initializer/random_uniform/min: (Const): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.222883: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/oweights/Initializer/random_uniform/min: (Const)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/oweights/Initializer/random_uniform/shape: (Const): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.222914: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/oweights/Initializer/random_uniform/shape: (Const)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/ibiases/Initializer/zeros: (Const): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.222940: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/ibiases/Initializer/zeros: (Const)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/iweights/Initializer/random_uniform/max: (Const): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.222962: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/iweights/Initializer/random_uniform/max: (Const)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/iweights/Initializer/random_uniform/min: (Const): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.222990: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/iweights/Initializer/random_uniform/min: (Const)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/iweights/Initializer/random_uniform/shape: (Const): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.223014: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/iweights/Initializer/random_uniform/shape: (Const)/job:localhost/replica:0/task:0/device:CPU:0
    fp32_storage/Placeholder: (Placeholder): /job:localhost/replica:0/task:0/device:CPU:0
    2018-10-11 19:19:15.223039: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/Placeholder: (Placeholder)/job:localhost/replica:0/task:0/device:CPU:0
    [   1] Compute Time (avg: 5.8425, sd: 0.0000) Loss: 0.0000
    [   2] Compute Time (avg: 5.4254, sd: 0.0000) Loss: 0.0000
    [   3] Compute Time (avg: 5.4253, sd: 0.0001) Loss: 0.0000
    [   4] Compute Time (avg: 5.3903, sd: 0.0495) Loss: 0.0000
    [   5] Compute Time (avg: 5.3953, sd: 0.0437) Loss: 0.0000
    [   6] Compute Time (avg: 5.3439, sd: 0.1098) Loss: 0.0000
    [   7] Compute Time (avg: 5.3158, sd: 0.1183) Loss: 0.0000
    [   8] Compute Time (avg: 5.3184, sd: 0.1097) Loss: 0.0000
    [   9] Compute Time (avg: 5.3200, sd: 0.1027) Loss: 0.0000
    [  10] Compute Time (avg: 5.3176, sd: 0.0971) Loss: 0.0000

## GPU

    root@19ea79fd8130:/workspace/MixedPrecision# mnist-full-tf --shape 3 256 256 --gpu
                        batch_size: 256
                 static_loss_scale: 1
                               gpu: True
                              half: False
                             shape: [3, 256, 256]
                            epochs: 10
                       hidden_size: 64
                          momentum: 0.9
                              prof: None
                           workers: 4
                       kernel_size: 3
                                lr: 0.1
                              data: None
                      weight_decay: 0.0001
                dynamic_loss_scale: False
                              fake: False
                           permute: False
                        print_freq: 10
                        hidden_num: 1
                          conv_num: 32
    2018-10-11 19:21:36.153624: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1356] Found device 0 with properties: 
    name: Tesla V100-PCIE-16GB major: 7 minor: 0 memoryClockRate(GHz): 1.38
    pciBusID: 0000:03:00.0
    totalMemory: 15.78GiB freeMemory: 15.37GiB
    2018-10-11 19:21:36.339670: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1356] Found device 1 with properties: 
    name: Tesla M40 major: 5 minor: 2 memoryClockRate(GHz): 1.112
    pciBusID: 0000:82:00.0
    totalMemory: 11.18GiB freeMemory: 11.06GiB
    2018-10-11 19:21:36.339738: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1435] Adding visible gpu devices: 0, 1
    2018-10-11 19:21:36.953437: I tensorflow/core/common_runtime/gpu/gpu_device.cc:923] Device interconnect StreamExecutor with strength 1 edge matrix:
    2018-10-11 19:21:36.953512: I tensorflow/core/common_runtime/gpu/gpu_device.cc:929]      0 1 
    2018-10-11 19:21:36.953531: I tensorflow/core/common_runtime/gpu/gpu_device.cc:942] 0:   N N 
    2018-10-11 19:21:36.953543: I tensorflow/core/common_runtime/gpu/gpu_device.cc:942] 1:   N N 
    2018-10-11 19:21:36.954241: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1053] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 14883 MB memory) -> physical GPU (device: 0, name: Tesla V100-PCIE-16GB, pci bus id: 0000:03:00.0, compute capability: 7.0)
    2018-10-11 19:21:37.278172: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1053] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:1 with 10710 MB memory) -> physical GPU (device: 1, name: Tesla M40, pci bus id: 0000:82:00.0, compute capability: 5.2)
    Device mapping:
    /job:localhost/replica:0/task:0/device:XLA_CPU:0 -> device: XLA_CPU device
    /job:localhost/replica:0/task:0/device:XLA_GPU:0 -> device: XLA_GPU device
    /job:localhost/replica:0/task:0/device:GPU:0 -> device: 0, name: Tesla V100-PCIE-16GB, pci bus id: 0000:03:00.0, compute capability: 7.0
    /job:localhost/replica:0/task:0/device:GPU:1 -> device: 1, name: Tesla M40, pci bus id: 0000:82:00.0, compute capability: 5.2
    2018-10-11 19:21:37.474459: I tensorflow/core/common_runtime/direct_session.cc:284] Device mapping:
    /job:localhost/replica:0/task:0/device:XLA_CPU:0 -> device: XLA_CPU device
    /job:localhost/replica:0/task:0/device:XLA_GPU:0 -> device: XLA_GPU device
    /job:localhost/replica:0/task:0/device:GPU:0 -> device: 0, name: Tesla V100-PCIE-16GB, pci bus id: 0000:03:00.0, compute capability: 7.0
    /job:localhost/replica:0/task:0/device:GPU:1 -> device: 1, name: Tesla M40, pci bus id: 0000:82:00.0, compute capability: 5.2
    
    fp32_storage/fp32_storage/obiases/Momentum: (VariableV2): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.479899: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/fp32_storage/obiases/Momentum: (VariableV2)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/fp32_storage/obiases/Momentum/read: (Identity): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.479943: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/fp32_storage/obiases/Momentum/read: (Identity)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/fp32_storage/obiases/Momentum/Assign: (Assign): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.479960: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/fp32_storage/obiases/Momentum/Assign: (Assign)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/fp32_storage/oweights/Momentum: (VariableV2): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.479976: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/fp32_storage/oweights/Momentum: (VariableV2)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/fp32_storage/oweights/Momentum/read: (Identity): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.479989: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/fp32_storage/oweights/Momentum/read: (Identity)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/fp32_storage/oweights/Momentum/Assign: (Assign): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.480002: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/fp32_storage/oweights/Momentum/Assign: (Assign)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/fp32_storage/ibiases/Momentum: (VariableV2): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.480015: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/fp32_storage/ibiases/Momentum: (VariableV2)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/fp32_storage/ibiases/Momentum/read: (Identity): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.480028: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/fp32_storage/ibiases/Momentum/read: (Identity)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/fp32_storage/ibiases/Momentum/Assign: (Assign): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.480041: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/fp32_storage/ibiases/Momentum/Assign: (Assign)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/fp32_storage/iweights/Momentum: (VariableV2): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.480056: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/fp32_storage/iweights/Momentum: (VariableV2)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/fp32_storage/iweights/Momentum/read: (Identity): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.480069: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/fp32_storage/iweights/Momentum/read: (Identity)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/fp32_storage/iweights/Momentum/Initializer/zeros: (Fill): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.480082: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/fp32_storage/iweights/Momentum/Initializer/zeros: (Fill)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/fp32_storage/iweights/Momentum/Assign: (Assign): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.480095: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/fp32_storage/iweights/Momentum/Assign: (Assign)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/input_layer/add_grad/BroadcastGradientArgs: (BroadcastGradientArgs): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.480110: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/input_layer/add_grad/BroadcastGradientArgs: (BroadcastGradientArgs)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/output_layer/add_grad/BroadcastGradientArgs: (BroadcastGradientArgs): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.480126: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/output_layer/add_grad/BroadcastGradientArgs: (BroadcastGradientArgs)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Mul_grad/BroadcastGradientArgs: (BroadcastGradientArgs): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.480141: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Mul_grad/BroadcastGradientArgs: (BroadcastGradientArgs)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/div_grad/BroadcastGradientArgs: (BroadcastGradientArgs): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.480156: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/div_grad/BroadcastGradientArgs: (BroadcastGradientArgs)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/Fill: (Fill): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.480171: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/Fill: (Fill)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/mul_grad/Mul: (Mul): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.480186: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/mul_grad/Mul: (Mul)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success: (NoOp): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.480201: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success: (NoOp)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/ones_like: (Fill): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.480217: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/ones_like: (Fill)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success: (NoOp): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.480233: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success: (NoOp)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/num_present/broadcast_weights/ones_like: (Fill): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.480248: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/num_present/broadcast_weights/ones_like: (Fill)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/num_present/ones_like: (Fill): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.480263: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/num_present/ones_like: (Fill)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/num_present/Equal: (Equal): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.480277: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/num_present/Equal: (Equal)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/num_present/Select: (Select): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.480295: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/num_present/Select: (Select)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/num_present/broadcast_weights: (Mul): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.480312: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/num_present/broadcast_weights: (Mul)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/num_present: (Sum): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.480331: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/num_present: (Sum)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/Equal: (Equal): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.480348: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/Equal: (Equal)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/Select: (Select): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.480365: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/Select: (Select)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/Greater: (Greater): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.480385: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/Greater: (Greater)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/value_grad/Select_1: (Select): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.480404: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/value_grad/Select_1: (Select)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/value_grad/Select: (Select): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.480419: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/value_grad/Select: (Select)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/div_grad/RealDiv: (RealDiv): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.480438: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/div_grad/RealDiv: (RealDiv)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/div_grad/Sum: (Sum): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.480457: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/div_grad/Sum: (Sum)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/div_grad/Reshape: (Reshape): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.480476: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/div_grad/Reshape: (Reshape)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Sum_1_grad/Reshape: (Reshape): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.480488: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Sum_1_grad/Reshape: (Reshape)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Sum_1_grad/Tile: (Tile): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.480498: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Sum_1_grad/Tile: (Tile)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Sum_grad/Reshape: (Reshape): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.480509: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Sum_grad/Reshape: (Reshape)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Sum_grad/Tile: (Tile): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.480518: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Sum_grad/Tile: (Tile)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Mul_grad/Mul: (Mul): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.480529: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Mul_grad/Mul: (Mul)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Mul_grad/Sum: (Sum): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.480539: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Mul_grad/Sum: (Sum)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Mul_grad/Reshape: (Reshape): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.480549: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Mul_grad/Reshape: (Reshape)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/xentropy/Reshape_2_grad/Reshape: (Reshape): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.480561: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/xentropy/Reshape_2_grad/Reshape: (Reshape)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/xentropy_grad/ExpandDims_1: (ExpandDims): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.480572: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/xentropy_grad/ExpandDims_1: (ExpandDims)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/xentropy_grad/ExpandDims: (ExpandDims): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.480583: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/xentropy_grad/ExpandDims: (ExpandDims)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/xentropy/Sub_1: (Sub): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.480604: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/xentropy/Sub_1: (Sub)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/xentropy/Slice_1/begin: (Pack): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.480622: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/xentropy/Slice_1/begin: (Pack)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/xentropy/Slice_1: (Slice): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.480635: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/xentropy/Slice_1: (Slice)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/xentropy/concat_1: (ConcatV2): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.480645: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/xentropy/concat_1: (ConcatV2)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/xentropy/Sub: (Sub): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.480660: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/xentropy/Sub: (Sub)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/xentropy/Slice/begin: (Pack): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.480681: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/xentropy/Slice/begin: (Pack)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/xentropy/Slice: (Slice): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.480700: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/xentropy/Slice: (Slice)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/xentropy/concat: (ConcatV2): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.480717: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/xentropy/concat: (ConcatV2)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/xentropy/Sub_2: (Sub): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.480732: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/xentropy/Sub_2: (Sub)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/xentropy/Slice_2/size: (Pack): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.480747: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/xentropy/Slice_2/size: (Pack)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/xentropy/Slice_2: (Slice): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.480763: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/xentropy/Slice_2: (Slice)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/labels_stop_gradient: (StopGradient): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.480779: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/labels_stop_gradient: (StopGradient)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/xentropy/Reshape_1: (Reshape): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.480796: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/xentropy/Reshape_1: (Reshape)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/obiases: (VariableV2): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.480811: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/obiases: (VariableV2)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/obiases/read: (Identity): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.480823: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/obiases/read: (Identity)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/obiases/Assign: (Assign): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.480836: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/obiases/Assign: (Assign)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/oweights: (VariableV2): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.480848: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/oweights: (VariableV2)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/oweights/read: (Identity): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.480860: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/oweights/read: (Identity)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/oweights/Initializer/random_uniform/sub: (Sub): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.480876: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/oweights/Initializer/random_uniform/sub: (Sub)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/oweights/Initializer/random_uniform/RandomUniform: (RandomUniform): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.480891: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/oweights/Initializer/random_uniform/RandomUniform: (RandomUniform)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/oweights/Initializer/random_uniform/mul: (Mul): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.480905: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/oweights/Initializer/random_uniform/mul: (Mul)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/oweights/Initializer/random_uniform: (Add): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.480920: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/oweights/Initializer/random_uniform: (Add)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/oweights/Assign: (Assign): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.480933: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/oweights/Assign: (Assign)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/ibiases: (VariableV2): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.480947: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/ibiases: (VariableV2)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/ibiases/read: (Identity): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.480956: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/ibiases/read: (Identity)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/ibiases/Assign: (Assign): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.480965: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/ibiases/Assign: (Assign)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/iweights: (VariableV2): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.480979: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/iweights: (VariableV2)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/iweights/read: (Identity): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.480994: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/iweights/read: (Identity)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/iweights/Initializer/random_uniform/sub: (Sub): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.481010: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/iweights/Initializer/random_uniform/sub: (Sub)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/iweights/Initializer/random_uniform/RandomUniform: (RandomUniform): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.481023: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/iweights/Initializer/random_uniform/RandomUniform: (RandomUniform)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/iweights/Initializer/random_uniform/mul: (Mul): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.481032: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/iweights/Initializer/random_uniform/mul: (Mul)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/iweights/Initializer/random_uniform: (Add): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.481041: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/iweights/Initializer/random_uniform: (Add)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/iweights/Assign: (Assign): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.481049: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/iweights/Assign: (Assign)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/init: (NoOp): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.481060: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/init: (NoOp)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/input_layer/MatMul: (MatMul): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.481073: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/input_layer/MatMul: (MatMul)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/input_layer/add: (Add): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.481089: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/input_layer/add: (Add)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/input_layer/Relu: (Relu): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.481104: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/input_layer/Relu: (Relu)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/output_layer/MatMul: (MatMul): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.481119: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/output_layer/MatMul: (MatMul)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/output_layer/add: (Add): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.481128: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/output_layer/add: (Add)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/xentropy/Reshape: (Reshape): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.481140: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/xentropy/Reshape: (Reshape)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/xentropy_grad/LogSoftmax: (LogSoftmax): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.481151: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/xentropy_grad/LogSoftmax: (LogSoftmax)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/xentropy_grad/Neg: (Neg): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.481165: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/xentropy_grad/Neg: (Neg)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/xentropy_grad/mul_1: (Mul): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.481183: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/xentropy_grad/mul_1: (Mul)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/xentropy: (SoftmaxCrossEntropyWithLogits): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.481199: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/xentropy: (SoftmaxCrossEntropyWithLogits)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/xentropy_grad/mul: (Mul): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.481210: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/xentropy_grad/mul: (Mul)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/xentropy/Reshape_grad/Reshape: (Reshape): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.481221: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/xentropy/Reshape_grad/Reshape: (Reshape)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/output_layer/add_grad/Sum_1: (Sum): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.481232: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/output_layer/add_grad/Sum_1: (Sum)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/output_layer/add_grad/Reshape_1: (Reshape): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.481256: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/output_layer/add_grad/Reshape_1: (Reshape)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/truediv_3: (RealDiv): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.481272: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/truediv_3: (RealDiv)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/Momentum/update_fp32_storage/obiases/ApplyMomentum: (ApplyMomentum): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.481289: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/Momentum/update_fp32_storage/obiases/ApplyMomentum: (ApplyMomentum)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/output_layer/add_grad/Sum: (Sum): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.481304: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/output_layer/add_grad/Sum: (Sum)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/output_layer/add_grad/Reshape: (Reshape): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.481321: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/output_layer/add_grad/Reshape: (Reshape)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/output_layer/MatMul_grad/MatMul_1: (MatMul): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.481339: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/output_layer/MatMul_grad/MatMul_1: (MatMul)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/truediv_2: (RealDiv): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.481357: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/truediv_2: (RealDiv)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/Momentum/update_fp32_storage/oweights/ApplyMomentum: (ApplyMomentum): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.481374: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/Momentum/update_fp32_storage/oweights/ApplyMomentum: (ApplyMomentum)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/output_layer/MatMul_grad/MatMul: (MatMul): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.481393: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/output_layer/MatMul_grad/MatMul: (MatMul)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/input_layer/Relu_grad/ReluGrad: (ReluGrad): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.481411: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/input_layer/Relu_grad/ReluGrad: (ReluGrad)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/input_layer/add_grad/Sum_1: (Sum): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.481427: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/input_layer/add_grad/Sum_1: (Sum)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/input_layer/add_grad/Reshape_1: (Reshape): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.481444: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/input_layer/add_grad/Reshape_1: (Reshape)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/truediv_1: (RealDiv): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.481459: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/truediv_1: (RealDiv)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/Momentum/update_fp32_storage/ibiases/ApplyMomentum: (ApplyMomentum): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.481469: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/Momentum/update_fp32_storage/ibiases/ApplyMomentum: (ApplyMomentum)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/input_layer/add_grad/Sum: (Sum): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.481479: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/input_layer/add_grad/Sum: (Sum)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/input_layer/add_grad/Reshape: (Reshape): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.481489: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/input_layer/add_grad/Reshape: (Reshape)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/input_layer/MatMul_grad/MatMul_1: (MatMul): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.481502: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/input_layer/MatMul_grad/MatMul_1: (MatMul)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/truediv: (RealDiv): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.481519: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/truediv: (RealDiv)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/Momentum/update_fp32_storage/iweights/ApplyMomentum: (ApplyMomentum): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.481536: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/Momentum/update_fp32_storage/iweights/ApplyMomentum: (ApplyMomentum)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/Momentum: (NoOp): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.481548: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/Momentum: (NoOp)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/input_layer/MatMul_grad/MatMul: (MatMul): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.481559: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/input_layer/MatMul_grad/MatMul: (MatMul)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/zeros_like: (ZerosLike): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.481569: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/zeros_like: (ZerosLike)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/xentropy/Reshape_2: (Reshape): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.481579: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/xentropy/Reshape_2: (Reshape)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Mul_grad/Mul_1: (Mul): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.481593: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Mul_grad/Mul_1: (Mul)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Mul_grad/Sum_1: (Sum): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.481604: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Mul_grad/Sum_1: (Sum)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Mul_grad/Reshape_1: (Reshape): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.481620: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Mul_grad/Reshape_1: (Reshape)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/Mul: (Mul): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.481635: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/Mul: (Mul)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/Sum: (Sum): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.481654: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/Sum: (Sum)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/Sum_1: (Sum): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.481671: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/Sum_1: (Sum)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/div_grad/Neg: (Neg): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.481689: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/div_grad/Neg: (Neg)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/div_grad/RealDiv_1: (RealDiv): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.481706: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/div_grad/RealDiv_1: (RealDiv)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/div_grad/RealDiv_2: (RealDiv): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.481724: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/div_grad/RealDiv_2: (RealDiv)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/div_grad/mul: (Mul): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.481742: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/div_grad/mul: (Mul)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/div_grad/Sum_1: (Sum): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.481757: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/div_grad/Sum_1: (Sum)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/div_grad/Reshape_1: (Reshape): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.481768: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/div_grad/Reshape_1: (Reshape)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/div: (RealDiv): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.481779: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/div: (RealDiv)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/value: (Select): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.481790: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/value: (Select)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/mul_grad/Mul_1: (Mul): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.481802: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/mul_grad/Mul_1: (Mul)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/mul: (Mul): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.481820: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/mul: (Mul)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/Momentum/momentum: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.481839: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/Momentum/momentum: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/Momentum/learning_rate: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.481858: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/Momentum/learning_rate: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/fp32_storage/obiases/Momentum/Initializer/zeros: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.481874: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/fp32_storage/obiases/Momentum/Initializer/zeros: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/fp32_storage/oweights/Momentum/Initializer/zeros: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.481889: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/fp32_storage/oweights/Momentum/Initializer/zeros: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/fp32_storage/ibiases/Momentum/Initializer/zeros: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.481904: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/fp32_storage/ibiases/Momentum/Initializer/zeros: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/fp32_storage/iweights/Momentum/Initializer/zeros/Const: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.481920: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/fp32_storage/iweights/Momentum/Initializer/zeros/Const: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/fp32_storage/iweights/Momentum/Initializer/zeros/shape_as_tensor: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.481936: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/fp32_storage/iweights/Momentum/Initializer/zeros/shape_as_tensor: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/truediv_3/y: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.481955: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/truediv_3/y: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/truediv_2/y: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.481972: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/truediv_2/y: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/truediv_1/y: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.481989: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/truediv_1/y: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/truediv/y: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.482006: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/truediv/y: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/input_layer/add_grad/Shape_1: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.482022: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/input_layer/add_grad/Shape_1: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/input_layer/add_grad/Shape: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.482040: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/input_layer/add_grad/Shape: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/output_layer/add_grad/Shape_1: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.482058: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/output_layer/add_grad/Shape_1: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/output_layer/add_grad/Shape: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.482074: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/output_layer/add_grad/Shape: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/xentropy/Reshape_grad/Shape: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.482091: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/xentropy/Reshape_grad/Shape: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/xentropy_grad/ExpandDims_1/dim: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.482108: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/xentropy_grad/ExpandDims_1/dim: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/xentropy_grad/ExpandDims/dim: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.482127: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/xentropy_grad/ExpandDims/dim: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/xentropy/Reshape_2_grad/Shape: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.482147: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/xentropy/Reshape_2_grad/Shape: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Mul_grad/Shape_1: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.482165: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Mul_grad/Shape_1: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Mul_grad/Shape: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.482182: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Mul_grad/Shape: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Sum_grad/Const: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.482199: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Sum_grad/Const: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Sum_grad/Reshape/shape: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.482216: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Sum_grad/Reshape/shape: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Sum_1_grad/Const: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.482233: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Sum_1_grad/Const: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Sum_1_grad/Reshape/shape: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.482249: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Sum_1_grad/Reshape/shape: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/div_grad/Shape_1: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.482261: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/div_grad/Shape_1: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/div_grad/Shape: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.482271: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/div_grad/Shape: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/value_grad/zeros_like: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.482282: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/value_grad/zeros_like: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/grad_ys_0: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.482296: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/grad_ys_0: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/Shape: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.482311: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/Shape: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/mul/y: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.482327: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/mul/y: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/zeros_like: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.482351: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/zeros_like: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/ones_like/Const: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.482364: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/ones_like/Const: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/ones_like/Shape: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.482375: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/ones_like/Shape: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/Equal/y: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.482385: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/Equal/y: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/Greater/y: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.482397: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/Greater/y: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/Const_1: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.482414: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/Const_1: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/num_present/Const: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.482429: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/num_present/Const: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/Const: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.482444: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/Const: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/Shape: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.482462: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/Shape: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/values/rank: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.482478: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/values/rank: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/values/shape: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.482494: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/values/shape: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/weights/rank: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.482506: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/weights/rank: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/weights/shape: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.482517: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/weights/shape: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/num_present/ones_like/Const: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.482528: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/num_present/ones_like/Const: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/num_present/ones_like/Shape: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.482540: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/num_present/ones_like/Shape: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/num_present/zeros_like: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.482555: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/num_present/zeros_like: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/num_present/Equal/y: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.482567: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/num_present/Equal/y: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/Const: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.482583: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/Const: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/ToFloat_1/x: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.482598: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/ToFloat_1/x: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/assert_broadcastable/values/rank: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.482610: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/assert_broadcastable/values/rank: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/assert_broadcastable/values/shape: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.482621: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/assert_broadcastable/values/shape: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/assert_broadcastable/weights/rank: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.482633: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/assert_broadcastable/weights/rank: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/assert_broadcastable/weights/shape: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.482646: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/assert_broadcastable/weights/shape: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/assert_broadcastable/weights: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.482666: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/assert_broadcastable/weights: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/xentropy/Slice_2/begin: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.482686: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/xentropy/Slice_2/begin: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/xentropy/Sub_2/y: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.482703: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/xentropy/Sub_2/y: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/xentropy/concat_1/axis: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.482720: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/xentropy/concat_1/axis: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/xentropy/concat_1/values_0: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.482737: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/xentropy/concat_1/values_0: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/xentropy/Slice_1/size: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.482748: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/xentropy/Slice_1/size: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/xentropy/Sub_1/y: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.482758: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/xentropy/Sub_1/y: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/xentropy/Shape_2: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.482770: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/xentropy/Shape_2: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/xentropy/Rank_2: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.482782: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/xentropy/Rank_2: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/xentropy/concat/axis: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.482799: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/xentropy/concat/axis: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/xentropy/concat/values_0: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.482818: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/xentropy/concat/values_0: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/xentropy/Slice/size: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.482833: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/xentropy/Slice/size: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/xentropy/Sub/y: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.482843: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/xentropy/Sub/y: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/xentropy/Shape_1: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.482854: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/xentropy/Shape_1: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/xentropy/Rank_1: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.482864: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/xentropy/Rank_1: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/xentropy/Shape: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.482876: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/xentropy/Shape: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/xentropy/Rank: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.482892: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/xentropy/Rank: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/Placeholder_1: (Placeholder): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.482908: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/Placeholder_1: (Placeholder)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/obiases/Initializer/zeros: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.482917: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/obiases/Initializer/zeros: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/oweights/Initializer/random_uniform/max: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.482926: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/oweights/Initializer/random_uniform/max: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/oweights/Initializer/random_uniform/min: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.482934: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/oweights/Initializer/random_uniform/min: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/oweights/Initializer/random_uniform/shape: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.482942: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/oweights/Initializer/random_uniform/shape: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/ibiases/Initializer/zeros: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.482954: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/ibiases/Initializer/zeros: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/iweights/Initializer/random_uniform/max: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.482968: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/iweights/Initializer/random_uniform/max: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/iweights/Initializer/random_uniform/min: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.482984: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/iweights/Initializer/random_uniform/min: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/iweights/Initializer/random_uniform/shape: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.482995: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/iweights/Initializer/random_uniform/shape: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/Placeholder: (Placeholder): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:21:37.483005: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/Placeholder: (Placeholder)/job:localhost/replica:0/task:0/device:GPU:0
    [   1] Compute Time (avg: 4.7514, sd: 0.0000) Loss: 0.0000
    [   2] Compute Time (avg: 4.2239, sd: 0.0000) Loss: 0.0000
    [   3] Compute Time (avg: 4.3249, sd: 0.1010) Loss: 0.0000
    [   4] Compute Time (avg: 4.3504, sd: 0.0900) Loss: 0.0000
    [   5] Compute Time (avg: 4.2962, sd: 0.1219) Loss: 0.0000
    [   6] Compute Time (avg: 4.3136, sd: 0.1145) Loss: 0.0000
    [   7] Compute Time (avg: 4.2969, sd: 0.1110) Loss: 0.0000
    [   8] Compute Time (avg: 4.2126, sd: 0.2306) Loss: 0.0000
    [   9] Compute Time (avg: 4.1426, sd: 0.2843) Loss: 0.0000
    [  10] Compute Time (avg: 4.1370, sd: 0.2685) Loss: 0.0000

## GPU half

    root@19ea79fd8130:/workspace/MixedPrecision# mnist-full-tf --shape 3 256 256 --gpu --half --lr 0.00001
                        hidden_num: 1
                dynamic_loss_scale: False
                          conv_num: 32
                           workers: 4
                            epochs: 10
                          momentum: 0.9
                       kernel_size: 3
                        print_freq: 10
                 static_loss_scale: 1
                              half: True
                           permute: False
                              prof: None
                              fake: False
                             shape: [3, 256, 256]
                      weight_decay: 0.0001
                       hidden_size: 64
                                lr: 1e-05
                        batch_size: 256
                               gpu: True
                              data: None
    2018-10-11 19:29:31.497989: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1356] Found device 0 with properties: 
    name: Tesla V100-PCIE-16GB major: 7 minor: 0 memoryClockRate(GHz): 1.38
    pciBusID: 0000:03:00.0
    totalMemory: 15.78GiB freeMemory: 15.37GiB
    2018-10-11 19:29:31.644005: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1356] Found device 1 with properties: 
    name: Tesla M40 major: 5 minor: 2 memoryClockRate(GHz): 1.112
    pciBusID: 0000:82:00.0
    totalMemory: 11.18GiB freeMemory: 11.06GiB
    2018-10-11 19:29:31.644059: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1435] Adding visible gpu devices: 0, 1
    2018-10-11 19:29:32.251238: I tensorflow/core/common_runtime/gpu/gpu_device.cc:923] Device interconnect StreamExecutor with strength 1 edge matrix:
    2018-10-11 19:29:32.251302: I tensorflow/core/common_runtime/gpu/gpu_device.cc:929]      0 1 
    2018-10-11 19:29:32.251311: I tensorflow/core/common_runtime/gpu/gpu_device.cc:942] 0:   N N 
    2018-10-11 19:29:32.251317: I tensorflow/core/common_runtime/gpu/gpu_device.cc:942] 1:   N N 
    2018-10-11 19:29:32.252017: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1053] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 14883 MB memory) -> physical GPU (device: 0, name: Tesla V100-PCIE-16GB, pci bus id: 0000:03:00.0, compute capability: 7.0)
    2018-10-11 19:29:32.497275: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1053] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:1 with 10710 MB memory) -> physical GPU (device: 1, name: Tesla M40, pci bus id: 0000:82:00.0, compute capability: 5.2)
    Device mapping:
    /job:localhost/replica:0/task:0/device:XLA_CPU:0 -> device: XLA_CPU device
    /job:localhost/replica:0/task:0/device:XLA_GPU:0 -> device: XLA_GPU device
    /job:localhost/replica:0/task:0/device:GPU:0 -> device: 0, name: Tesla V100-PCIE-16GB, pci bus id: 0000:03:00.0, compute capability: 7.0
    /job:localhost/replica:0/task:0/device:GPU:1 -> device: 1, name: Tesla M40, pci bus id: 0000:82:00.0, compute capability: 5.2
    2018-10-11 19:29:32.783127: I tensorflow/core/common_runtime/direct_session.cc:284] Device mapping:
    /job:localhost/replica:0/task:0/device:XLA_CPU:0 -> device: XLA_CPU device
    /job:localhost/replica:0/task:0/device:XLA_GPU:0 -> device: XLA_GPU device
    /job:localhost/replica:0/task:0/device:GPU:0 -> device: 0, name: Tesla V100-PCIE-16GB, pci bus id: 0000:03:00.0, compute capability: 7.0
    /job:localhost/replica:0/task:0/device:GPU:1 -> device: 1, name: Tesla M40, pci bus id: 0000:82:00.0, compute capability: 5.2
    
    fp32_storage/fp32_storage/obiases/Momentum: (VariableV2): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.791502: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/fp32_storage/obiases/Momentum: (VariableV2)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/fp32_storage/obiases/Momentum/read: (Identity): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.791560: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/fp32_storage/obiases/Momentum/read: (Identity)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/fp32_storage/obiases/Momentum/Assign: (Assign): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.791585: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/fp32_storage/obiases/Momentum/Assign: (Assign)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/fp32_storage/oweights/Momentum: (VariableV2): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.791608: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/fp32_storage/oweights/Momentum: (VariableV2)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/fp32_storage/oweights/Momentum/read: (Identity): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.791631: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/fp32_storage/oweights/Momentum/read: (Identity)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/fp32_storage/oweights/Momentum/Assign: (Assign): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.791651: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/fp32_storage/oweights/Momentum/Assign: (Assign)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/fp32_storage/ibiases/Momentum: (VariableV2): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.791672: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/fp32_storage/ibiases/Momentum: (VariableV2)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/fp32_storage/ibiases/Momentum/read: (Identity): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.791696: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/fp32_storage/ibiases/Momentum/read: (Identity)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/fp32_storage/ibiases/Momentum/Assign: (Assign): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.791730: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/fp32_storage/ibiases/Momentum/Assign: (Assign)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/fp32_storage/iweights/Momentum: (VariableV2): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.791764: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/fp32_storage/iweights/Momentum: (VariableV2)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/fp32_storage/iweights/Momentum/read: (Identity): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.791896: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/fp32_storage/iweights/Momentum/read: (Identity)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/fp32_storage/iweights/Momentum/Initializer/zeros: (Fill): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.791993: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/fp32_storage/iweights/Momentum/Initializer/zeros: (Fill)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/fp32_storage/iweights/Momentum/Assign: (Assign): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.792116: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/fp32_storage/iweights/Momentum/Assign: (Assign)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/input_layer/add_grad/BroadcastGradientArgs: (BroadcastGradientArgs): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.792166: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/input_layer/add_grad/BroadcastGradientArgs: (BroadcastGradientArgs)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/output_layer/add_grad/BroadcastGradientArgs: (BroadcastGradientArgs): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.792202: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/output_layer/add_grad/BroadcastGradientArgs: (BroadcastGradientArgs)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Mul_grad/BroadcastGradientArgs: (BroadcastGradientArgs): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.792245: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Mul_grad/BroadcastGradientArgs: (BroadcastGradientArgs)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/div_grad/BroadcastGradientArgs: (BroadcastGradientArgs): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.792287: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/div_grad/BroadcastGradientArgs: (BroadcastGradientArgs)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/Fill: (Fill): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.792325: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/Fill: (Fill)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/mul_grad/Mul: (Mul): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.792362: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/mul_grad/Mul: (Mul)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success: (NoOp): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.792399: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success: (NoOp)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/ones_like: (Fill): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.792437: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/ones_like: (Fill)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success: (NoOp): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.792476: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success: (NoOp)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/num_present/broadcast_weights/ones_like: (Fill): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.792501: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/num_present/broadcast_weights/ones_like: (Fill)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/num_present/ones_like: (Fill): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.792524: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/num_present/ones_like: (Fill)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/num_present/Equal: (Equal): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.792560: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/num_present/Equal: (Equal)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/num_present/Select: (Select): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.792600: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/num_present/Select: (Select)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/num_present/broadcast_weights: (Mul): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.792643: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/num_present/broadcast_weights: (Mul)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/num_present: (Sum): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.792679: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/num_present: (Sum)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/Equal: (Equal): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.792709: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/Equal: (Equal)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/Select: (Select): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.792745: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/Select: (Select)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/Greater: (Greater): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.792781: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/Greater: (Greater)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/value_grad/Select_1: (Select): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.792822: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/value_grad/Select_1: (Select)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/value_grad/Select: (Select): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.792859: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/value_grad/Select: (Select)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/div_grad/RealDiv: (RealDiv): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.792897: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/div_grad/RealDiv: (RealDiv)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/div_grad/Sum: (Sum): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.792935: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/div_grad/Sum: (Sum)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/div_grad/Reshape: (Reshape): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.792971: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/div_grad/Reshape: (Reshape)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Sum_1_grad/Reshape: (Reshape): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.793009: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Sum_1_grad/Reshape: (Reshape)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Sum_1_grad/Tile: (Tile): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.793046: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Sum_1_grad/Tile: (Tile)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Sum_grad/Reshape: (Reshape): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.793086: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Sum_grad/Reshape: (Reshape)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Sum_grad/Tile: (Tile): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.793124: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Sum_grad/Tile: (Tile)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Mul_grad/Mul: (Mul): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.793159: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Mul_grad/Mul: (Mul)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Mul_grad/Sum: (Sum): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.793195: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Mul_grad/Sum: (Sum)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Mul_grad/Reshape: (Reshape): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.793228: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Mul_grad/Reshape: (Reshape)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/xentropy/Reshape_2_grad/Reshape: (Reshape): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.793262: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/xentropy/Reshape_2_grad/Reshape: (Reshape)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/xentropy_grad/ExpandDims_1: (ExpandDims): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.793291: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/xentropy_grad/ExpandDims_1: (ExpandDims)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/xentropy_grad/ExpandDims: (ExpandDims): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.793328: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/xentropy_grad/ExpandDims: (ExpandDims)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/xentropy/Sub_1: (Sub): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.793366: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/xentropy/Sub_1: (Sub)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/xentropy/Slice_1/begin: (Pack): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.793404: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/xentropy/Slice_1/begin: (Pack)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/xentropy/Slice_1: (Slice): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.793439: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/xentropy/Slice_1: (Slice)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/xentropy/concat_1: (ConcatV2): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.793475: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/xentropy/concat_1: (ConcatV2)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/xentropy/Sub: (Sub): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.793527: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/xentropy/Sub: (Sub)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/xentropy/Slice/begin: (Pack): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.793566: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/xentropy/Slice/begin: (Pack)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/xentropy/Slice: (Slice): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.793605: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/xentropy/Slice: (Slice)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/xentropy/concat: (ConcatV2): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.793641: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/xentropy/concat: (ConcatV2)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/xentropy/Sub_2: (Sub): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.793678: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/xentropy/Sub_2: (Sub)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/xentropy/Slice_2/size: (Pack): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.793705: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/xentropy/Slice_2/size: (Pack)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/xentropy/Slice_2: (Slice): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.793739: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/xentropy/Slice_2: (Slice)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/labels_stop_gradient: (StopGradient): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.793777: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/labels_stop_gradient: (StopGradient)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/xentropy/Reshape_1: (Reshape): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.793813: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/xentropy/Reshape_1: (Reshape)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/obiases: (VariableV2): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.793849: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/obiases: (VariableV2)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/obiases/read: (Identity): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.793877: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/obiases/read: (Identity)/job:localhost/replica:0/task:0/device:GPU:0
    Cast_3: (Cast): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.793912: I tensorflow/core/common_runtime/placer.cc:886] Cast_3: (Cast)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/obiases/Assign: (Assign): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.793936: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/obiases/Assign: (Assign)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/oweights: (VariableV2): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.793957: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/oweights: (VariableV2)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/oweights/read: (Identity): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.793983: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/oweights/read: (Identity)/job:localhost/replica:0/task:0/device:GPU:0
    Cast_2: (Cast): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.794051: I tensorflow/core/common_runtime/placer.cc:886] Cast_2: (Cast)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/oweights/Initializer/random_uniform/sub: (Sub): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.794088: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/oweights/Initializer/random_uniform/sub: (Sub)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/oweights/Initializer/random_uniform/RandomUniform: (RandomUniform): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.794122: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/oweights/Initializer/random_uniform/RandomUniform: (RandomUniform)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/oweights/Initializer/random_uniform/mul: (Mul): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.794155: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/oweights/Initializer/random_uniform/mul: (Mul)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/oweights/Initializer/random_uniform: (Add): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.794184: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/oweights/Initializer/random_uniform: (Add)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/oweights/Assign: (Assign): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.794215: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/oweights/Assign: (Assign)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/ibiases: (VariableV2): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.794247: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/ibiases: (VariableV2)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/ibiases/read: (Identity): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.794279: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/ibiases/read: (Identity)/job:localhost/replica:0/task:0/device:GPU:0
    Cast_1: (Cast): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.794314: I tensorflow/core/common_runtime/placer.cc:886] Cast_1: (Cast)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/ibiases/Assign: (Assign): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.794358: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/ibiases/Assign: (Assign)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/iweights: (VariableV2): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.794390: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/iweights: (VariableV2)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/iweights/read: (Identity): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.794410: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/iweights/read: (Identity)/job:localhost/replica:0/task:0/device:GPU:0
    Cast: (Cast): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.794435: I tensorflow/core/common_runtime/placer.cc:886] Cast: (Cast)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/iweights/Initializer/random_uniform/sub: (Sub): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.794466: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/iweights/Initializer/random_uniform/sub: (Sub)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/iweights/Initializer/random_uniform/RandomUniform: (RandomUniform): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.794499: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/iweights/Initializer/random_uniform/RandomUniform: (RandomUniform)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/iweights/Initializer/random_uniform/mul: (Mul): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.794544: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/iweights/Initializer/random_uniform/mul: (Mul)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/iweights/Initializer/random_uniform: (Add): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.794573: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/iweights/Initializer/random_uniform: (Add)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/iweights/Assign: (Assign): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.794604: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/iweights/Assign: (Assign)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/init: (NoOp): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.794635: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/init: (NoOp)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/input_layer/MatMul: (MatMul): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.794671: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/input_layer/MatMul: (MatMul)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/input_layer/add: (Add): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.794703: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/input_layer/add: (Add)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/input_layer/Relu: (Relu): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.794740: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/input_layer/Relu: (Relu)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/output_layer/MatMul: (MatMul): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.794772: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/output_layer/MatMul: (MatMul)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/output_layer/add: (Add): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.794803: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/output_layer/add: (Add)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/Cast: (Cast): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.794834: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/Cast: (Cast)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/xentropy/Reshape: (Reshape): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.794868: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/xentropy/Reshape: (Reshape)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/xentropy_grad/LogSoftmax: (LogSoftmax): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.794900: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/xentropy_grad/LogSoftmax: (LogSoftmax)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/xentropy_grad/Neg: (Neg): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.794930: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/xentropy_grad/Neg: (Neg)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/xentropy_grad/mul_1: (Mul): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.794963: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/xentropy_grad/mul_1: (Mul)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/xentropy: (SoftmaxCrossEntropyWithLogits): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.794997: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/xentropy: (SoftmaxCrossEntropyWithLogits)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/xentropy_grad/mul: (Mul): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.795028: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/xentropy_grad/mul: (Mul)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/xentropy/Reshape_grad/Reshape: (Reshape): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.795087: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/xentropy/Reshape_grad/Reshape: (Reshape)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/Cast_grad/Cast: (Cast): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.795123: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/Cast_grad/Cast: (Cast)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/output_layer/add_grad/Sum_1: (Sum): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.795156: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/output_layer/add_grad/Sum_1: (Sum)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/output_layer/add_grad/Reshape_1: (Reshape): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.795190: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/output_layer/add_grad/Reshape_1: (Reshape)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/Cast_3_grad/Cast: (Cast): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.795224: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/Cast_3_grad/Cast: (Cast)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/truediv_3: (RealDiv): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.795258: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/truediv_3: (RealDiv)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/Momentum/update_fp32_storage/obiases/ApplyMomentum: (ApplyMomentum): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.795290: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/Momentum/update_fp32_storage/obiases/ApplyMomentum: (ApplyMomentum)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/output_layer/add_grad/Sum: (Sum): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.795323: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/output_layer/add_grad/Sum: (Sum)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/output_layer/add_grad/Reshape: (Reshape): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.795357: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/output_layer/add_grad/Reshape: (Reshape)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/output_layer/MatMul_grad/MatMul_1: (MatMul): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.795391: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/output_layer/MatMul_grad/MatMul_1: (MatMul)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/Cast_2_grad/Cast: (Cast): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.795424: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/Cast_2_grad/Cast: (Cast)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/truediv_2: (RealDiv): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.795486: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/truediv_2: (RealDiv)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/Momentum/update_fp32_storage/oweights/ApplyMomentum: (ApplyMomentum): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.795525: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/Momentum/update_fp32_storage/oweights/ApplyMomentum: (ApplyMomentum)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/output_layer/MatMul_grad/MatMul: (MatMul): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.795560: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/output_layer/MatMul_grad/MatMul: (MatMul)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/input_layer/Relu_grad/ReluGrad: (ReluGrad): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.795593: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/input_layer/Relu_grad/ReluGrad: (ReluGrad)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/input_layer/add_grad/Sum_1: (Sum): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.795627: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/input_layer/add_grad/Sum_1: (Sum)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/input_layer/add_grad/Reshape_1: (Reshape): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.795668: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/input_layer/add_grad/Reshape_1: (Reshape)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/Cast_1_grad/Cast: (Cast): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.795704: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/Cast_1_grad/Cast: (Cast)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/truediv_1: (RealDiv): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.795737: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/truediv_1: (RealDiv)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/Momentum/update_fp32_storage/ibiases/ApplyMomentum: (ApplyMomentum): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.795768: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/Momentum/update_fp32_storage/ibiases/ApplyMomentum: (ApplyMomentum)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/input_layer/add_grad/Sum: (Sum): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.795799: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/input_layer/add_grad/Sum: (Sum)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/input_layer/add_grad/Reshape: (Reshape): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.795832: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/input_layer/add_grad/Reshape: (Reshape)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/input_layer/MatMul_grad/MatMul_1: (MatMul): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.795864: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/input_layer/MatMul_grad/MatMul_1: (MatMul)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/Cast_grad/Cast: (Cast): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.795896: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/Cast_grad/Cast: (Cast)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/truediv: (RealDiv): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.795929: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/truediv: (RealDiv)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/Momentum/update_fp32_storage/iweights/ApplyMomentum: (ApplyMomentum): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.795959: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/Momentum/update_fp32_storage/iweights/ApplyMomentum: (ApplyMomentum)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/Momentum: (NoOp): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.795993: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/Momentum: (NoOp)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/input_layer/MatMul_grad/MatMul: (MatMul): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.796028: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/input_layer/MatMul_grad/MatMul: (MatMul)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/zeros_like: (ZerosLike): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.796063: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/zeros_like: (ZerosLike)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/xentropy/Reshape_2: (Reshape): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.796094: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/xentropy/Reshape_2: (Reshape)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Mul_grad/Mul_1: (Mul): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.796124: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Mul_grad/Mul_1: (Mul)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Mul_grad/Sum_1: (Sum): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.796176: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Mul_grad/Sum_1: (Sum)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Mul_grad/Reshape_1: (Reshape): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.796220: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Mul_grad/Reshape_1: (Reshape)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/Mul: (Mul): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.796253: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/Mul: (Mul)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/Sum: (Sum): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.796286: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/Sum: (Sum)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/Sum_1: (Sum): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.796316: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/Sum_1: (Sum)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/div_grad/Neg: (Neg): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.796350: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/div_grad/Neg: (Neg)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/div_grad/RealDiv_1: (RealDiv): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.796381: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/div_grad/RealDiv_1: (RealDiv)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/div_grad/RealDiv_2: (RealDiv): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.796413: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/div_grad/RealDiv_2: (RealDiv)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/div_grad/mul: (Mul): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.796441: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/div_grad/mul: (Mul)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/div_grad/Sum_1: (Sum): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.796473: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/div_grad/Sum_1: (Sum)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/div_grad/Reshape_1: (Reshape): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.796509: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/div_grad/Reshape_1: (Reshape)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/div: (RealDiv): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.796547: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/div: (RealDiv)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/value: (Select): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.796583: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/value: (Select)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/mul_grad/Mul_1: (Mul): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.796615: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/mul_grad/Mul_1: (Mul)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/mul: (Mul): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.796648: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/mul: (Mul)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/Momentum/momentum: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.796683: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/Momentum/momentum: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/Momentum/learning_rate: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.796717: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/Momentum/learning_rate: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/fp32_storage/obiases/Momentum/Initializer/zeros: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.796759: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/fp32_storage/obiases/Momentum/Initializer/zeros: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/fp32_storage/oweights/Momentum/Initializer/zeros: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.796794: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/fp32_storage/oweights/Momentum/Initializer/zeros: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/fp32_storage/ibiases/Momentum/Initializer/zeros: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.796824: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/fp32_storage/ibiases/Momentum/Initializer/zeros: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/fp32_storage/iweights/Momentum/Initializer/zeros/Const: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.796867: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/fp32_storage/iweights/Momentum/Initializer/zeros/Const: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/fp32_storage/iweights/Momentum/Initializer/zeros/shape_as_tensor: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.796901: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/fp32_storage/iweights/Momentum/Initializer/zeros/shape_as_tensor: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/truediv_3/y: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.796933: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/truediv_3/y: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/truediv_2/y: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.796969: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/truediv_2/y: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/truediv_1/y: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.797002: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/truediv_1/y: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/truediv/y: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.797057: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/truediv/y: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/input_layer/add_grad/Shape_1: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.797097: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/input_layer/add_grad/Shape_1: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/input_layer/add_grad/Shape: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.797135: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/input_layer/add_grad/Shape: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/output_layer/add_grad/Shape_1: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.797167: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/output_layer/add_grad/Shape_1: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/output_layer/add_grad/Shape: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.797197: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/output_layer/add_grad/Shape: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/xentropy/Reshape_grad/Shape: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.797233: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/xentropy/Reshape_grad/Shape: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/xentropy_grad/ExpandDims_1/dim: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.797261: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/xentropy_grad/ExpandDims_1/dim: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/xentropy_grad/ExpandDims/dim: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.797285: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/xentropy_grad/ExpandDims/dim: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/xentropy/Reshape_2_grad/Shape: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.797318: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/xentropy/Reshape_2_grad/Shape: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Mul_grad/Shape_1: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.797353: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Mul_grad/Shape_1: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Mul_grad/Shape: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.797386: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Mul_grad/Shape: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Sum_grad/Const: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.797442: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Sum_grad/Const: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Sum_grad/Reshape/shape: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.797482: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Sum_grad/Reshape/shape: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Sum_1_grad/Const: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.797519: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Sum_1_grad/Const: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Sum_1_grad/Reshape/shape: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.797554: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/Sum_1_grad/Reshape/shape: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/div_grad/Shape_1: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.797590: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/div_grad/Shape_1: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/div_grad/Shape: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.797620: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/div_grad/Shape: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/value_grad/zeros_like: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.797654: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/fp32_storage/softmax_cross_entropy_loss/value_grad/zeros_like: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/grad_ys_0: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.797688: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/grad_ys_0: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/gradients/Shape: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.797721: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/gradients/Shape: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/mul/y: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.797774: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/mul/y: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/zeros_like: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.797811: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/zeros_like: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/ones_like/Const: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.797849: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/ones_like/Const: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/ones_like/Shape: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.797883: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/ones_like/Shape: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/Equal/y: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.797917: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/Equal/y: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/Greater/y: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.797975: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/Greater/y: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/Const_1: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.798012: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/Const_1: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/num_present/Const: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.798048: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/num_present/Const: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/Const: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.798083: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/Const: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/Shape: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.798118: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/Shape: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/values/rank: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.798153: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/values/rank: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/values/shape: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.798188: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/values/shape: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/weights/rank: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.798223: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/weights/rank: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/weights/shape: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.798260: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/weights/shape: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/num_present/ones_like/Const: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.798299: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/num_present/ones_like/Const: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/num_present/ones_like/Shape: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.798346: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/num_present/ones_like/Shape: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/num_present/zeros_like: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.798400: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/num_present/zeros_like: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/num_present/Equal/y: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.798439: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/num_present/Equal/y: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/Const: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.798473: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/Const: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/ToFloat_1/x: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.798506: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/ToFloat_1/x: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/assert_broadcastable/values/rank: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.798540: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/assert_broadcastable/values/rank: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/assert_broadcastable/values/shape: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.798572: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/assert_broadcastable/values/shape: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/assert_broadcastable/weights/rank: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.798606: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/assert_broadcastable/weights/rank: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/assert_broadcastable/weights/shape: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.798641: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/assert_broadcastable/weights/shape: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/assert_broadcastable/weights: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.798673: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/assert_broadcastable/weights: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/xentropy/Slice_2/begin: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.798709: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/xentropy/Slice_2/begin: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/xentropy/Sub_2/y: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.798744: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/xentropy/Sub_2/y: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/xentropy/concat_1/axis: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.798777: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/xentropy/concat_1/axis: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/xentropy/concat_1/values_0: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.798812: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/xentropy/concat_1/values_0: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/xentropy/Slice_1/size: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.798846: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/xentropy/Slice_1/size: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/xentropy/Sub_1/y: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.798880: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/xentropy/Sub_1/y: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/xentropy/Shape_2: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.798912: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/xentropy/Shape_2: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/xentropy/Rank_2: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.798951: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/xentropy/Rank_2: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/xentropy/concat/axis: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.798989: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/xentropy/concat/axis: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/xentropy/concat/values_0: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.799030: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/xentropy/concat/values_0: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/xentropy/Slice/size: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.799066: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/xentropy/Slice/size: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/xentropy/Sub/y: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.799128: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/xentropy/Sub/y: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/xentropy/Shape_1: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.799165: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/xentropy/Shape_1: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/xentropy/Rank_1: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.799197: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/xentropy/Rank_1: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/xentropy/Shape: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.799233: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/xentropy/Shape: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/softmax_cross_entropy_loss/xentropy/Rank: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.799270: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/softmax_cross_entropy_loss/xentropy/Rank: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/Placeholder_1: (Placeholder): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.799300: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/Placeholder_1: (Placeholder)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/obiases/Initializer/zeros: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.799333: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/obiases/Initializer/zeros: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/oweights/Initializer/random_uniform/max: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.799364: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/oweights/Initializer/random_uniform/max: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/oweights/Initializer/random_uniform/min: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.799407: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/oweights/Initializer/random_uniform/min: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/oweights/Initializer/random_uniform/shape: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.799445: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/oweights/Initializer/random_uniform/shape: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/ibiases/Initializer/zeros: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.799475: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/ibiases/Initializer/zeros: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/iweights/Initializer/random_uniform/max: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.799506: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/iweights/Initializer/random_uniform/max: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/iweights/Initializer/random_uniform/min: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.799537: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/iweights/Initializer/random_uniform/min: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/iweights/Initializer/random_uniform/shape: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.799566: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/iweights/Initializer/random_uniform/shape: (Const)/job:localhost/replica:0/task:0/device:GPU:0
    fp32_storage/Placeholder: (Placeholder): /job:localhost/replica:0/task:0/device:GPU:0
    2018-10-11 19:29:32.799601: I tensorflow/core/common_runtime/placer.cc:886] fp32_storage/Placeholder: (Placeholder)/job:localhost/replica:0/task:0/device:GPU:0
    [   1] Compute Time (avg: 1.6187, sd: 0.0000) Loss: 0.0000
    [   2] Compute Time (avg: 0.9699, sd: 0.0000) Loss: 0.0000
    [   3] Compute Time (avg: 0.9028, sd: 0.0672) Loss: 0.0000
    [   4] Compute Time (avg: 0.8489, sd: 0.0938) Loss: 0.0000
    [   5] Compute Time (avg: 0.8671, sd: 0.0871) Loss: 0.0000
    [   6] Compute Time (avg: 0.9134, sd: 0.1211) Loss: 0.0000
    [   7] Compute Time (avg: 0.8917, sd: 0.1207) Loss: 0.0000
    [   8] Compute Time (avg: 0.8763, sd: 0.1180) Loss: 0.0000
    [   9] Compute Time (avg: 0.8615, sd: 0.1172) Loss: 0.0000
    [  10] Compute Time (avg: 0.8743, sd: 0.1163) Loss: 0.0000
