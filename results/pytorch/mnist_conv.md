# MNIST Convolution (784 x 1024 x Conv(out_channel=512, kernel_size=3) x 10)


## `mnist-conv --data /data/ --hidden_size 1024 --conv_num 512 --gpu`

### Output

    [   1] Compute Time (avg: 10.4352, sd: 0.0000) Loss: 2.3026
    [   2] Compute Time (avg: 9.2671, sd: 0.0000) Loss: 2.3026
    [   3] Compute Time (avg: 9.2739, sd: 0.0068) Loss: 2.3026
    [   4] Compute Time (avg: 9.2978, sd: 0.0342) Loss: 2.3026
    [   5] Compute Time (avg: 9.3102, sd: 0.0366) Loss: 2.3026
    [   6] Compute Time (avg: 9.3616, sd: 0.1079) Loss: 2.3026
    [   7] Compute Time (avg: 9.3833, sd: 0.1098) Loss: 2.3026
    [   8] Compute Time (avg: 9.3855, sd: 0.1018) Loss: 2.3026
    [   9] Compute Time (avg: 9.4000, sd: 0.1026) Loss: 2.3026
    [  10] Compute Time (avg: 9.2112, sd: 0.5428) Loss: 2.302

### nvprof

## `mnist-conv --data /data/ --hidden_size 1024 --conv_num 512 --gpu --half`

### Output

    [   1] Compute Time (avg: 9.8191, sd: 0.0000) Loss: 2.3027
    [   2] Compute Time (avg: 8.5427, sd: 0.0000) Loss: 2.3027
    [   3] Compute Time (avg: 8.5728, sd: 0.0301) Loss: 2.3027
    [   4] Compute Time (avg: 8.5566, sd: 0.0336) Loss: 2.3027
    [   5] Compute Time (avg: 8.5351, sd: 0.0472) Loss: 2.3027
    [   6] Compute Time (avg: 8.5334, sd: 0.0424) Loss: 2.3027
    [   7] Compute Time (avg: 8.5327, sd: 0.0387) Loss: 2.3027
    [   8] Compute Time (avg: 8.5204, sd: 0.0468) Loss: 2.3027
    [   9] Compute Time (avg: 8.5286, sd: 0.0489) Loss: 2.3027
    [  10] Compute Time (avg: 8.5142, sd: 0.0616) Loss: 2.3027

### nvprof

## Summary

1.08 x speed up