# MNIST Fully connected network (784 x 16384 x 10)

## `mnist-full --data /data/ --hidden_size 16384 --gpu`

### Output

    [   1] Compute Time (avg: 20.4450, sd: 0.0000) Loss: 1.6176
    [   2] Compute Time (avg: 19.9917, sd: 0.0000) Loss: 1.4857
    [   3] Compute Time (avg: 20.1151, sd: 0.1235) Loss: 1.5031
    [   4] Compute Time (avg: 20.1336, sd: 0.1041) Loss: 1.4948
    [   5] Compute Time (avg: 20.1050, sd: 0.1029) Loss: 1.4847
    [   6] Compute Time (avg: 20.1031, sd: 0.0921) Loss: 1.4765
    [   7] Compute Time (avg: 20.1362, sd: 0.1120) Loss: 1.4680
    
### nvprof

    Type   Time(%)   Time   Calls   Avg   Min   Max   Name
                    %   s    ms   ms   ms   
    GPU activities   26.330835   4.380366    236    18.560872  0.331774    37.205433   volta_sgemm_128x64_tn
    GPU activities   17.982992   2.991629    470    6.365167   0.030048    23.069017   volta_sgemm_128x64_nt
    GPU activities   17.696788   2.944016    234   12.581265   9.810741    22.925977   volta_sgemm_128x128_nn
    GPU activities   14.843802   2.469397   2814    0.877539   0.001152     9.29167    void kernelPointwiseApply2<TensorAddOp<float>,
    GPU activities   14.748605   2.45356    2820    0.870056   0.001152     9.233143   void kernelPointwiseApply2<TensorCAddOp<float>, f
    GPU activities   5.099526    0.848351   1410    0.601667   0.001216     8.195675   kernelPointwiseApply1<TensorMulConstantOp<float>,
    GPU activities   0.845486    0.140654    235    0.598527   0.282334     3.184882   volta_sgemm_64x32_sliced1x4_nt
    GPU activities   0.819946    0.136405    477    0.285964   0.001024   114.962621   [CUDA memcpy HtoD]
    GPU activities   0.817271    0.13596     234    0.581026   0.54691      0.617405   volta_sgemm_128x128_tn
    GPU activities   0.177887    0.029593    470    0.062963   0.023903     0.0768     void kernelPointwiseApply3<ThresholdUpdateGradI
    GPU activities   0.169205    0.028149    235    0.119781   0.071584     2.715156   volta_sgemm_32x32_sliced1x4_tn
    GPU activities   0.121302    0.02018     470    0.042935   0.015552     0.045568   void kernelPointwiseApply2<TensorMaxValueOp<flo
    
 
 ## `mnist-full --data /data/ --hidden_size 16384 --gpu --half`
 

### Output
 
    [   1] Compute Time (avg: 8.2159, sd: 0.0000) Loss: 1.6172
    [   2] Compute Time (avg: 7.9991, sd: 0.0000) Loss: 1.4863
    [   3] Compute Time (avg: 7.9769, sd: 0.0223) Loss: 1.5010
    [   4] Compute Time (avg: 7.9070, sd: 0.1004) Loss: 1.5059
    [   5] Compute Time (avg: 7.9052, sd: 0.0870) Loss: 1.4893
    [   6] Compute Time (avg: 7.9301, sd: 0.0924) Loss: 1.4717
    [   7] Compute Time (avg: 7.9362, sd: 0.0854) Loss: 1.4707
    
### nvprof

    Type   Time(%)   Time   Calls   Avg   Min   Max   Name
        %   s   ms   ms   ms   
    GPU activities   23.340944   1.418031   2820   0.502847   0.001279     7.28755   kernelPointwiseApply2<TensorCAddOp<__h
    GPU activities   22.818796   1.386309   2814   0.492646   0.001215     7.289311  kernelPointwiseApply2<TensorAddOp<
    GPU activities   17.920284   1.08871     468   2.326302   0.166271     6.49181   volta_fp16_s884gemm_fp16_64x128_ldg8_f2f_tn
    GPU activities   12.264137   0.745082    470   1.585281   0.074624     4.985737  volta_fp16_s884gemm_fp16_128x128_ldg8_f2f_nt
    GPU activities    9.768179   0.593445    234   2.536091   1.765944     4.34414   volta_fp16_s884gemm_fp16_256x128_ldg8_f2f_nn
    GPU activities    9.59593    0.582981   1410   0.413461   0.001184     4.269932  kernelPointwiseApply1<TensorMulConstantOp<__half>,
    GPU activities    2.263622   0.137522    477   0.288305   0.001056   113.633177  [CUDA memcpy HtoD]
    GPU activities    0.406878   0.024719    235   0.105187   0.046527     0.105855  volta_fp16_sgemm_fp16_128x128_nt
    GPU activities    0.317442   0.019286    470   0.041033   0.0176       0.052608  kernelPointwiseApply3<ThresholdUpdateGradInp
    
# Result

2.5 x Speed up