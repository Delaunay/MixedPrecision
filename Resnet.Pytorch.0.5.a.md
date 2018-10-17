

    nvidia-docker run --ipc=host -it -v /Tmp/delaunap:/data --rm nvcr.io/nvidia/pytorch:18.07-py3
    
GPU: Tesla V100-PCIE-16GB

# Resnet 18 - Single

## Vanilla

    [   1][  10] Batch Time (avg: 0.1152, sd: 0.0000) Speed (avg: 1111.3039) Data (avg: 0.0015, sd: 0.0016)
    [   1][  20] Batch Time (avg: 0.1127, sd: 0.0014) Speed (avg: 1135.9902) Data (avg: 0.0141, sd: 0.0363)
    [   1][  30] Batch Time (avg: 0.1121, sd: 0.0017) Speed (avg: 1141.5845) Data (avg: 0.0127, sd: 0.0314)
    [   1][  40] Batch Time (avg: 0.1119, sd: 0.0015) Speed (avg: 1143.8783) Data (avg: 0.0096, sd: 0.0276)
    [   1][  50] Batch Time (avg: 0.1119, sd: 0.0014) Speed (avg: 1143.7080) Data (avg: 0.0109, sd: 0.0286)
    [   1][  60] Batch Time (avg: 0.1117, sd: 0.0015) Speed (avg: 1145.4447) Data (avg: 0.0106, sd: 0.0282)
    [   1][  70] Batch Time (avg: 0.1116, sd: 0.0014) Speed (avg: 1146.5412) Data (avg: 0.0158, sd: 0.0410)
    [   1][  80] Batch Time (avg: 0.1116, sd: 0.0014) Speed (avg: 1146.8259) Data (avg: 0.0191, sd: 0.0493)
    [   1][  90] Batch Time (avg: 0.1117, sd: 0.0014) Speed (avg: 1146.0777) Data (avg: 0.0256, sd: 0.0641)
    [   1][ 100] Batch Time (avg: 0.1117, sd: 0.0013) Speed (avg: 1145.6808) Data (avg: 0.0277, sd: 0.0678)
                 Metric      Average   Deviation          Min          Max
    0  CPU Data loading     0.033690   0.0946648     0.000082     0.708659
    1  GPU Data Loading     0.007753  0.00219666     0.004963     0.020880
    2  Waiting for data     0.027743   0.0678201     0.000372     0.321469
    3  CPU Compute Time     0.111724  0.00134826     0.109627     0.116119
    4  GPU Compute Time     0.111426  0.00116254     0.109557     0.115290
    5   Full Batch Time     0.142114   0.0710639     0.110219     0.432938
    6     Compute Speed  1145.680829          NA  1102.320588  1167.598024
    7   Effective Speed   900.687324          NA   295.654601  1161.324254


## Dali
    
NVJPEG error "6"

        
# Resnet 18 - Half

## Vanilla

    [   1][  10] Batch Time (avg: 0.0557, sd: 0.0000) Speed (avg: 2299.7944) Data (avg: 0.0160, sd: 0.0416)
    [   1][  20] Batch Time (avg: 0.0549, sd: 0.0009) Speed (avg: 2333.5426) Data (avg: 0.0388, sd: 0.0651)
    [   1][  30] Batch Time (avg: 0.0550, sd: 0.0008) Speed (avg: 2326.7837) Data (avg: 0.0546, sd: 0.0769)
    [   1][  40] Batch Time (avg: 0.0553, sd: 0.0012) Speed (avg: 2316.2666) Data (avg: 0.0523, sd: 0.0754)
    [   1][  50] Batch Time (avg: 0.0551, sd: 0.0012) Speed (avg: 2322.5510) Data (avg: 0.0662, sd: 0.1136)
    [   1][  60] Batch Time (avg: 0.0551, sd: 0.0011) Speed (avg: 2323.7525) Data (avg: 0.0648, sd: 0.1138)
    [   1][  70] Batch Time (avg: 0.0550, sd: 0.0010) Speed (avg: 2328.3955) Data (avg: 0.0685, sd: 0.1181)
    [   1][  80] Batch Time (avg: 0.0549, sd: 0.0010) Speed (avg: 2330.5012) Data (avg: 0.0674, sd: 0.1176)
    [   1][  90] Batch Time (avg: 0.0550, sd: 0.0012) Speed (avg: 2326.6848) Data (avg: 0.0715, sd: 0.1221)
    [   1][ 100] Batch Time (avg: 0.0550, sd: 0.0012) Speed (avg: 2327.1657) Data (avg: 0.0748, sd: 0.1228)
                 Metric      Average   Deviation          Min          Max
    0  CPU Data loading     0.077992    0.133163     0.000097     0.627697
    1  GPU Data Loading     0.006282  0.00227517     0.002362     0.015659
    2  Waiting for data     0.074751    0.122846     0.000526     0.503216
    3  CPU Compute Time     0.055003  0.00117893     0.053382     0.059496
    4  GPU Compute Time     0.054797  0.00111719     0.053265     0.059164
    5   Full Batch Time     0.135648    0.126856     0.054063     0.557791
    6     Compute Speed  2327.165660          NA  2151.416437  2397.804887
    7   Effective Speed   943.619744          NA   229.476732  2367.604581

    
## Dali

NVJPEG error "6"


# Resnet 50 - Single
## Vanilla

    [   1][  10] Batch Time (avg: 0.3676, sd: 0.0000) Speed (avg: 348.1953) Data (avg: 0.0019, sd: 0.0036)
    [   1][  20] Batch Time (avg: 0.3632, sd: 0.0005) Speed (avg: 352.4598) Data (avg: 0.0012, sd: 0.0026)
    [   1][  30] Batch Time (avg: 0.3629, sd: 0.0011) Speed (avg: 352.7619) Data (avg: 0.0010, sd: 0.0021)
    [   1][  40] Batch Time (avg: 0.3631, sd: 0.0014) Speed (avg: 352.5680) Data (avg: 0.0010, sd: 0.0018)
    [   1][  50] Batch Time (avg: 0.3634, sd: 0.0015) Speed (avg: 352.2218) Data (avg: 0.0009, sd: 0.0016)
    [   1][  60] Batch Time (avg: 0.3633, sd: 0.0014) Speed (avg: 352.2876) Data (avg: 0.0009, sd: 0.0015)
    [   1][  70] Batch Time (avg: 0.3633, sd: 0.0013) Speed (avg: 352.3232) Data (avg: 0.0009, sd: 0.0014)
    [   1][  80] Batch Time (avg: 0.3633, sd: 0.0013) Speed (avg: 352.3327) Data (avg: 0.0009, sd: 0.0013)
    [   1][  90] Batch Time (avg: 0.3632, sd: 0.0012) Speed (avg: 352.4288) Data (avg: 0.0008, sd: 0.0012)
    [   1][ 100] Batch Time (avg: 0.3634, sd: 0.0014) Speed (avg: 352.2647) Data (avg: 0.0009, sd: 0.0012)
                 Metric     Average   Deviation         Min         Max
    0  CPU Data loading    0.006518   0.0642241    0.000101    0.651963
    1  GPU Data Loading    0.008174  0.00166517    0.007275    0.024067
    2  Waiting for data    0.000853  0.00115869    0.000503    0.012216
    3  CPU Compute Time    0.363363  0.00137978    0.360917    0.369214
    4  GPU Compute Time    0.363114  0.00132583    0.360780    0.369003
    5   Full Batch Time    0.364125  0.00143397    0.361488    0.370235
    6     Compute Speed  352.264734          NA  346.682357  354.651854
    7   Effective Speed  351.527195          NA  345.726172  354.091874

## Dali


    
# Resnet 50 - Half
## Vanila
    
    [   1][  10] Batch Time (avg: 0.1733, sd: 0.0000) Speed (avg: 738.4186) Data (avg: 0.0014, sd: 0.0021)
    [   1][  20] Batch Time (avg: 0.1707, sd: 0.0029) Speed (avg: 749.6449) Data (avg: 0.0011, sd: 0.0015)
    [   1][  30] Batch Time (avg: 0.1704, sd: 0.0024) Speed (avg: 751.3487) Data (avg: 0.0010, sd: 0.0012)
    [   1][  40] Batch Time (avg: 0.1700, sd: 0.0023) Speed (avg: 752.9796) Data (avg: 0.0010, sd: 0.0010)
    [   1][  50] Batch Time (avg: 0.1693, sd: 0.0026) Speed (avg: 756.2298) Data (avg: 0.0010, sd: 0.0009)
    [   1][  60] Batch Time (avg: 0.1691, sd: 0.0026) Speed (avg: 756.9286) Data (avg: 0.0009, sd: 0.0009)
    [   1][  70] Batch Time (avg: 0.1692, sd: 0.0026) Speed (avg: 756.5982) Data (avg: 0.0009, sd: 0.0008)
    [   1][  80] Batch Time (avg: 0.1694, sd: 0.0025) Speed (avg: 755.8058) Data (avg: 0.0009, sd: 0.0007)
    [   1][  90] Batch Time (avg: 0.1693, sd: 0.0026) Speed (avg: 755.8443) Data (avg: 0.0009, sd: 0.0007)
    [   1][ 100] Batch Time (avg: 0.1691, sd: 0.0026) Speed (avg: 757.0607) Data (avg: 0.0009, sd: 0.0007)
                 Metric     Average    Deviation         Min         Max
    0  CPU Data loading    0.006398    0.0627807    0.000099    0.637336
    1  GPU Data Loading    0.007910   0.00294417    0.002343    0.017783
    2  Waiting for data    0.000894  0.000674414    0.000548    0.007295
    3  CPU Compute Time    0.169075   0.00258256    0.164888    0.177987
    4  GPU Compute Time    0.168812   0.00253613    0.164702    0.177726
    5   Full Batch Time    0.169937   0.00266062    0.165521    0.178952
    6     Compute Speed  757.060730           NA  719.154211  776.286077
    7   Effective Speed  753.218180           NA  715.274737  773.313982
    
## Dali

