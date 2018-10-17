GPU: Tesla V100-PCIE-16GB

# Resnet 18 - Single

    > resnet-18-pt --gpu --data /Tmp/delaunap/img_net/ -b 128 -j 4 --static-loss-scale 128 --prof 10
    
    [   1][  10] Batch Time (avg: 0.1119, sd: 0.0000) Speed (avg: 1143.7167) Data (avg: 0.5213, sd: 1.0087)
    [   1][  20] Batch Time (avg: 0.1140, sd: 0.0049) Speed (avg: 1123.0181) Data (avg: 0.5646, sd: 0.9679)
    [   1][  30] Batch Time (avg: 0.1185, sd: 0.0117) Speed (avg: 1080.1146) Data (avg: 0.5004, sd: 0.9007)
    [   1][  40] Batch Time (avg: 0.1207, sd: 0.0130) Speed (avg: 1060.2788) Data (avg: 0.4980, sd: 0.8672)
    [   1][  50] Batch Time (avg: 0.1212, sd: 0.0139) Speed (avg: 1055.9874) Data (avg: 0.4825, sd: 0.8585)
    [   1][  60] Batch Time (avg: 0.1213, sd: 0.0132) Speed (avg: 1054.9497) Data (avg: 0.5149, sd: 0.8422)
    [   1][  70] Batch Time (avg: 0.1207, sd: 0.0126) Speed (avg: 1060.6552) Data (avg: 0.5041, sd: 0.8224)
    [   1][  80] Batch Time (avg: 0.1213, sd: 0.0128) Speed (avg: 1055.2177) Data (avg: 0.5270, sd: 0.8005)
    [   1][  90] Batch Time (avg: 0.1222, sd: 0.0130) Speed (avg: 1047.6154) Data (avg: 0.5204, sd: 0.7744)
    [   1][ 100] Batch Time (avg: 0.1220, sd: 0.0127) Speed (avg: 1048.8176) Data (avg: 0.5383, sd: 0.7595)
    Data Loading (CPU) (0.5492, sd: 0.5492, min: 0.0001, max: 2.9883)
    Data Loading (GPU) (0.0072, sd: 0.0072, min: 0.0023, max: 0.0426)
    Time waiting for data 0.5383, sd: 0.5383, min: 0.0006, max: 2.9400)
    CPU Compute Time(avg: 0.1220, sd: 0.1220, min: 0.1084, max: 0.1641) Speed 1048.8176
    GPU Compute Time(avg: 0.1171, sd: 0.1171, min: 0.1082, max: 0.1582) Speed 1092.9972
    Full Batch Time (avg: 0.6620, sd: 0.6620, min: 0.1091, max: 2.5801) Speed 193.3493
    [   1] 10 Batch Time (avg: 69.1687, sd: 0.0000)  Batch Time (max: 0.1641, min: 0.1084) Loss: 5.4411
    
# Resnet 18 - Half

    > resnet-18-pt --gpu --data /Tmp/delaunap/img_net/ -b 128 -j 4 --static-loss-scale 128 --prof 10 --half
    
    [   1][  10] Batch Time (avg: 0.0763, sd: 0.0000) Speed (avg: 1677.1660) Data (avg: 0.5351, sd: 0.9062)
    [   1][  20] Batch Time (avg: 0.0765, sd: 0.0071) Speed (avg: 1673.0470) Data (avg: 0.5860, sd: 0.9379)
    [   1][  30] Batch Time (avg: 0.0815, sd: 0.0104) Speed (avg: 1571.0521) Data (avg: 0.5203, sd: 0.8895)
    [   1][  40] Batch Time (avg: 0.0822, sd: 0.0236) Speed (avg: 1556.3387) Data (avg: 0.5335, sd: 0.8863)
    [   1][  50] Batch Time (avg: 0.0820, sd: 0.0210) Speed (avg: 1560.6580) Data (avg: 0.5111, sd: 0.8509)
    [   1][  60] Batch Time (avg: 0.0807, sd: 0.0198) Speed (avg: 1585.7889) Data (avg: 0.5402, sd: 0.8173)
    [   1][  70] Batch Time (avg: 0.0799, sd: 0.0190) Speed (avg: 1601.0638) Data (avg: 0.5421, sd: 0.7881)
    [   1][  80] Batch Time (avg: 0.0798, sd: 0.0180) Speed (avg: 1604.5800) Data (avg: 0.5552, sd: 0.7763)
    [   1][  90] Batch Time (avg: 0.0801, sd: 0.0175) Speed (avg: 1598.0545) Data (avg: 0.5553, sd: 0.7685)
    [   1][ 100] Batch Time (avg: 0.0800, sd: 0.0170) Speed (avg: 1599.0359) Data (avg: 0.5745, sd: 0.8147)
    Data Loading (CPU) (0.5844, sd: 0.5844, min: 0.0001, max: 2.8954)
    Data Loading (GPU) (0.0058, sd: 0.0058, min: 0.0022, max: 0.0297)
    Time waiting for data 0.5745, sd: 0.5745, min: 0.0006, max: 2.6051)
    CPU Compute Time(avg: 0.0800, sd: 0.0800, min: 0.0598, max: 0.1932) Speed 1599.0359
    GPU Compute Time(avg: 0.0750, sd: 0.0750, min: 0.0597, max: 0.1753) Speed 1705.5866
    Full Batch Time (avg: 0.6586, sd: 0.6586, min: 0.0607, max: 2.6651) Speed 194.3625
    [   1] 10 Batch Time (avg: 68.5609, sd: 0.0000)  Batch Time (max: 0.1932, min: 0.0598) Loss: 5.4219
    
# Resnet 50 - Single

    > resnet-50-pt --gpu --data /Tmp/delaunap/img_net/ -b 128 -j 4 --static-loss-scale 128 --prof 10
    
    [   1][  10] Batch Time (avg: 0.3965, sd: 0.0000) Speed (avg: 322.8015) Data (avg: 0.1813, sd: 0.4852)
    [   1][  20] Batch Time (avg: 0.3772, sd: 0.0108) Speed (avg: 339.3273) Data (avg: 0.2282, sd: 0.4465)
    [   1][  30] Batch Time (avg: 0.3765, sd: 0.0131) Speed (avg: 339.9710) Data (avg: 0.2322, sd: 0.4543)
    [   1][  40] Batch Time (avg: 0.3791, sd: 0.0139) Speed (avg: 337.6096) Data (avg: 0.2400, sd: 0.4399)
    [   1][  50] Batch Time (avg: 0.3788, sd: 0.0129) Speed (avg: 337.9072) Data (avg: 0.2625, sd: 0.4907)
    [   1][  60] Batch Time (avg: 0.3792, sd: 0.0133) Speed (avg: 337.5347) Data (avg: 0.2920, sd: 0.5281)
    [   1][  70] Batch Time (avg: 0.3792, sd: 0.0129) Speed (avg: 337.5182) Data (avg: 0.2962, sd: 0.5427)
    [   1][  80] Batch Time (avg: 0.3786, sd: 0.0130) Speed (avg: 338.0576) Data (avg: 0.3163, sd: 0.5662)
    [   1][  90] Batch Time (avg: 0.3784, sd: 0.0133) Speed (avg: 338.2647) Data (avg: 0.3136, sd: 0.5707)
    [   1][ 100] Batch Time (avg: 0.3779, sd: 0.0132) Speed (avg: 338.7048) Data (avg: 0.3224, sd: 0.5763)
    Data Loading (CPU) (0.3331, sd: 0.3331, min: 0.0001, max: 2.6096)
    Data Loading (GPU) (0.0096, sd: 0.0096, min: 0.0023, max: 0.0370)
    Time waiting for data 0.3224, sd: 0.3224, min: 0.0006, max: 1.8872)
    CPU Compute Time(avg: 0.3779, sd: 0.3779, min: 0.3561, max: 0.4195) Speed 338.7048
    GPU Compute Time(avg: 0.3698, sd: 0.3698, min: 0.3560, max: 0.4025) Speed 346.1207
    Full Batch Time (avg: 0.7145, sd: 0.7145, min: 0.3570, max: 2.3017) Speed 179.1521
    [   1] 10 Batch Time (avg: 75.0378, sd: 0.0000)  Batch Time (max: 0.4195, min: 0.3561) Loss: 5.4460
    
# Resnet 50 - Half

