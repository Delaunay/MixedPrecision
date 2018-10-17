GPU: Tesla V100-PCIE-16GB

# Resnet 18 - Single

## Vanilla

    > resnet-18-pt --gpu --data /Tmp/delaunap/img_net/ -b 128 -j 4 --static-loss-scale 128 --prof 10
    
    [   1][  10] Batch Time (avg: 0.1237, sd: 0.0000) Speed (avg: 1034.8740) Data (avg: 0.4814, sd: 0.7135)
    [   1][  20] Batch Time (avg: 0.1367, sd: 0.0127) Speed (avg: 936.0756) Data (avg: 0.5039, sd: 0.6878)
    [   1][  30] Batch Time (avg: 0.1311, sd: 0.0131) Speed (avg: 976.3941) Data (avg: 0.4716, sd: 0.5891)
    [   1][  40] Batch Time (avg: 0.1289, sd: 0.0148) Speed (avg: 993.1506) Data (avg: 0.4819, sd: 0.5398)
    [   1][  50] Batch Time (avg: 0.1277, sd: 0.0148) Speed (avg: 1002.0917) Data (avg: 0.4794, sd: 0.5452)
    [   1][  60] Batch Time (avg: 0.1266, sd: 0.0143) Speed (avg: 1011.3519) Data (avg: 0.5140, sd: 0.6520)
    [   1][  70] Batch Time (avg: 0.1262, sd: 0.0139) Speed (avg: 1014.0313) Data (avg: 0.5107, sd: 0.6518)
    [   1][  80] Batch Time (avg: 0.1267, sd: 0.0139) Speed (avg: 1009.9415) Data (avg: 0.5355, sd: 0.6572)
    [   1][  90] Batch Time (avg: 0.1254, sd: 0.0138) Speed (avg: 1020.8695) Data (avg: 0.5311, sd: 0.6511)
    [   1][ 100] Batch Time (avg: 0.1253, sd: 0.0140) Speed (avg: 1021.7515) Data (avg: 0.5392, sd: 0.6635)
                 Metric      Average   Deviation         Min          Max
    0  CPU Data loading     0.546677    0.692012    0.000065     2.647985
    1  GPU Data Loading     0.006863  0.00538249    0.002269     0.040263
    2  Waiting for data     0.539221    0.663527    0.000618     2.620503
    3  CPU Compute Time     0.125275   0.0139962    0.108425     0.165703
    4  GPU Compute Time     0.118853   0.0110803    0.108188     0.165589
    5   Full Batch Time     0.670300    0.659273    0.109384     2.743997
    6     Compute Speed  1021.751498          NA  772.467241  1180.543207
    7   Effective Speed   190.959399          NA   46.647276  1170.188784

## Dali
    
    [   1][  10] Batch Time (avg: 0.1252, sd: 0.0000) Speed (avg: 1022.6346) Data (avg: 0.0721, sd: 0.0468)
    [   1][  20] Batch Time (avg: 0.1436, sd: 0.0250) Speed (avg: 891.6158) Data (avg: 0.0931, sd: 0.0468)
    [   1][  30] Batch Time (avg: 0.1393, sd: 0.0210) Speed (avg: 918.6527) Data (avg: 0.0908, sd: 0.0396)
    [   1][  40] Batch Time (avg: 0.1393, sd: 0.0185) Speed (avg: 918.8931) Data (avg: 0.0830, sd: 0.0374)
    [   1][  50] Batch Time (avg: 0.1373, sd: 0.0173) Speed (avg: 932.3596) Data (avg: 0.0726, sd: 0.0404)
    [   1][  60] Batch Time (avg: 0.1381, sd: 0.0172) Speed (avg: 926.6663) Data (avg: 0.0720, sd: 0.0418)
    [   1][  70] Batch Time (avg: 0.1376, sd: 0.0167) Speed (avg: 930.3301) Data (avg: 0.0809, sd: 0.0469)
    [   1][  80] Batch Time (avg: 0.1371, sd: 0.0161) Speed (avg: 933.6179) Data (avg: 0.0815, sd: 0.0445)
    [   1][  90] Batch Time (avg: 0.1365, sd: 0.0160) Speed (avg: 937.3864) Data (avg: 0.0812, sd: 0.0437)
    [   1][ 100] Batch Time (avg: 0.1370, sd: 0.0156) Speed (avg: 934.0063) Data (avg: 0.0793, sd: 0.0422)
                 Metric     Average  Deviation         Min          Max
    0  CPU Data loading    0.000000          0         inf         -inf
    1  GPU Data Loading    0.000000          0         inf         -inf
    2  Waiting for data    0.079341  0.0421711    0.000121     0.216322
    3  CPU Compute Time    0.137044  0.0155853    0.108722     0.201213
    4  GPU Compute Time    0.132623  0.0154415    0.108579     0.201071
    5   Full Batch Time    0.217131  0.0453859    0.131086     0.363098
    6     Compute Speed  934.006321         NA  636.141416  1177.314927
    7   Effective Speed  589.504752         NA  352.521537   976.457376
        
# Resnet 18 - Half

## Vanilla

    > resnet-18-pt --gpu --data /Tmp/delaunap/img_net/ -b 128 -j 4 --static-loss-scale 128 --prof 10 --half
    
    [   1][  10] Batch Time (avg: 0.0713, sd: 0.0000) Speed (avg: 1794.3367) Data (avg: 0.5188, sd: 0.8485)
    [   1][  20] Batch Time (avg: 0.0760, sd: 0.0100) Speed (avg: 1684.4880) Data (avg: 0.5602, sd: 0.8708)
    [   1][  30] Batch Time (avg: 0.0808, sd: 0.0143) Speed (avg: 1584.0961) Data (avg: 0.5067, sd: 0.8433)
    [   1][  40] Batch Time (avg: 0.0835, sd: 0.0243) Speed (avg: 1532.5166) Data (avg: 0.5310, sd: 0.8437)
    [   1][  50] Batch Time (avg: 0.0837, sd: 0.0216) Speed (avg: 1528.7665) Data (avg: 0.5208, sd: 0.8674)
    [   1][  60] Batch Time (avg: 0.0830, sd: 0.0204) Speed (avg: 1541.2901) Data (avg: 0.5526, sd: 0.8634)
    [   1][  70] Batch Time (avg: 0.0820, sd: 0.0195) Speed (avg: 1560.8118) Data (avg: 0.5400, sd: 0.8693)
    [   1][  80] Batch Time (avg: 0.0811, sd: 0.0184) Speed (avg: 1578.4265) Data (avg: 0.5550, sd: 0.8496)
    [   1][  90] Batch Time (avg: 0.0796, sd: 0.0178) Speed (avg: 1607.1269) Data (avg: 0.5586, sd: 0.8121)
    [   1][ 100] Batch Time (avg: 0.0796, sd: 0.0169) Speed (avg: 1608.2012) Data (avg: 0.5681, sd: 0.7838)
                 Metric      Average   Deviation         Min          Max
    0  CPU Data loading     0.576613    0.809435    0.000072     2.833230
    1  GPU Data Loading     0.006528  0.00511576    0.002236     0.027743
    2  Waiting for data     0.568068    0.783829    0.000603     2.558556
    3  CPU Compute Time     0.079592   0.0169107    0.059691     0.195768
    4  GPU Compute Time     0.075483    0.013203    0.059574     0.172219
    5   Full Batch Time     0.652599    0.772992    0.060487     2.627091
    6     Compute Speed  1608.201238          NA  653.836351  2144.378588
    7   Effective Speed   196.138822          NA   48.723095  2116.147732
    
## Dali

    [   1][  10] Batch Time (avg: 0.0789, sd: 0.0000) Speed (avg: 1622.7411) Data (avg: 0.1178, sd: 0.0651)
    [   1][  20] Batch Time (avg: 0.0913, sd: 0.0114) Speed (avg: 1402.1526) Data (avg: 0.1259, sd: 0.0477)
    [   1][  30] Batch Time (avg: 0.0915, sd: 0.0117) Speed (avg: 1398.2337) Data (avg: 0.1210, sd: 0.0404)
    [   1][  40] Batch Time (avg: 0.0898, sd: 0.0107) Speed (avg: 1424.9146) Data (avg: 0.1105, sd: 0.0404)
    [   1][  50] Batch Time (avg: 0.0884, sd: 0.0123) Speed (avg: 1447.9606) Data (avg: 0.0979, sd: 0.0456)
    [   1][  60] Batch Time (avg: 0.0869, sd: 0.0135) Speed (avg: 1473.2283) Data (avg: 0.0985, sd: 0.0443)
    [   1][  70] Batch Time (avg: 0.0862, sd: 0.0133) Speed (avg: 1484.3762) Data (avg: 0.1081, sd: 0.0492)
    [   1][  80] Batch Time (avg: 0.0858, sd: 0.0131) Speed (avg: 1491.4565) Data (avg: 0.1088, sd: 0.0463)
    [   1][  90] Batch Time (avg: 0.0861, sd: 0.0136) Speed (avg: 1486.5557) Data (avg: 0.1082, sd: 0.0450)
    [   1][ 100] Batch Time (avg: 0.0856, sd: 0.0144) Speed (avg: 1496.0887) Data (avg: 0.1080, sd: 0.0430)
                 Metric      Average  Deviation          Min          Max
    0  CPU Data loading     0.000000          0          inf         -inf
    1  GPU Data Loading     0.000000          0          inf         -inf
    2  Waiting for data     0.107975  0.0430214     0.000094     0.240893
    3  CPU Compute Time     0.085556  0.0143703     0.059912     0.127898
    4  GPU Compute Time     0.079042  0.0126129     0.059629     0.120114
    5   Full Batch Time     0.192572  0.0425764     0.094687     0.332003
    6     Compute Speed  1496.088739         NA  1000.801415  2136.451015
    7   Effective Speed   664.687807         NA   385.538549  1351.819130
    
# Resnet 50 - Single

## Vanilla

    > resnet-50-pt --gpu --data /Tmp/delaunap/img_net/ -b 128 -j 4 --static-loss-scale 128 --prof 10
    
    [   1][  10] Batch Time (avg: 0.3735, sd: 0.0000) Speed (avg: 342.6981) Data (avg: 0.1751, sd: 0.4625)
    [   1][  20] Batch Time (avg: 0.3863, sd: 0.0157) Speed (avg: 331.3339) Data (avg: 0.2490, sd: 0.4821)
    [   1][  30] Batch Time (avg: 0.3783, sd: 0.0153) Speed (avg: 338.3738) Data (avg: 0.2300, sd: 0.4291)
    [   1][  40] Batch Time (avg: 0.3772, sd: 0.0147) Speed (avg: 339.3204) Data (avg: 0.2438, sd: 0.3945)
    [   1][  50] Batch Time (avg: 0.3749, sd: 0.0145) Speed (avg: 341.4325) Data (avg: 0.2542, sd: 0.3973)
    [   1][  60] Batch Time (avg: 0.3757, sd: 0.0148) Speed (avg: 340.6618) Data (avg: 0.2839, sd: 0.4415)
    [   1][  70] Batch Time (avg: 0.3767, sd: 0.0150) Speed (avg: 339.7715) Data (avg: 0.2805, sd: 0.4294)
    [   1][  80] Batch Time (avg: 0.3766, sd: 0.0147) Speed (avg: 339.8579) Data (avg: 0.3015, sd: 0.4254)
    [   1][  90] Batch Time (avg: 0.3767, sd: 0.0146) Speed (avg: 339.8060) Data (avg: 0.3030, sd: 0.4301)
    [   1][ 100] Batch Time (avg: 0.3768, sd: 0.0143) Speed (avg: 339.7405) Data (avg: 0.3196, sd: 0.4693)
                 Metric     Average   Deviation         Min         Max
    0  CPU Data loading    0.328002    0.503528    0.000075    2.224498
    1  GPU Data Loading    0.008808  0.00617137    0.002276    0.029739
    2  Waiting for data    0.319623    0.469342    0.000630    1.721456
    3  CPU Compute Time    0.376758   0.0142911    0.355746    0.409725
    4  GPU Compute Time    0.369561   0.0125455    0.355628    0.406157
    5   Full Batch Time    0.710849    0.467764    0.356435    2.109562
    6     Compute Speed  339.740457          NA  312.404881  359.807729
    7   Effective Speed  180.066413          NA   60.676091  359.111458

## Dali

OOM after 50 batchs
    
# Resnet 50 - Half


## Vanila

    > resnet-50-pt --gpu --data /Tmp/delaunap/img_net/ -b 128 -j 4 --static-loss-scale 128 --prof 10 --half
    
    [   1][  10] Batch Time (avg: 0.2259, sd: 0.0000) Speed (avg: 566.6064) Data (avg: 0.2988, sd: 0.6535)
    [   1][  20] Batch Time (avg: 0.2056, sd: 0.0229) Speed (avg: 622.6399) Data (avg: 0.4199, sd: 0.7491)
    [   1][  30] Batch Time (avg: 0.2015, sd: 0.0200) Speed (avg: 635.2218) Data (avg: 0.3713, sd: 0.6783)
    [   1][  40] Batch Time (avg: 0.2020, sd: 0.0179) Speed (avg: 633.7873) Data (avg: 0.3979, sd: 0.6894)
    [   1][  50] Batch Time (avg: 0.2049, sd: 0.0176) Speed (avg: 624.8078) Data (avg: 0.3912, sd: 0.6748)
    [   1][  60] Batch Time (avg: 0.2033, sd: 0.0166) Speed (avg: 629.6450) Data (avg: 0.4351, sd: 0.6798)
    [   1][  70] Batch Time (avg: 0.2030, sd: 0.0159) Speed (avg: 630.5124) Data (avg: 0.4278, sd: 0.6813)
    [   1][  80] Batch Time (avg: 0.2030, sd: 0.0152) Speed (avg: 630.4354) Data (avg: 0.4466, sd: 0.6722)
    [   1][  90] Batch Time (avg: 0.2032, sd: 0.0149) Speed (avg: 629.9705) Data (avg: 0.4451, sd: 0.6494)
    [   1][ 100] Batch Time (avg: 0.2028, sd: 0.0145) Speed (avg: 631.1789) Data (avg: 0.4592, sd: 0.6634)
                 Metric     Average   Deviation         Min         Max
    0  CPU Data loading    0.470661    0.700621    0.000076    2.875573
    1  GPU Data Loading    0.007058  0.00459732    0.003137    0.018926
    2  Waiting for data    0.459160     0.66341    0.000616    2.071677
    3  CPU Compute Time    0.202795   0.0145319    0.180202    0.264656
    4  GPU Compute Time    0.196527   0.0130744    0.179712    0.258855
    5   Full Batch Time    0.678003    0.661191    0.182765    2.191550
    6     Compute Speed  631.178908          NA  483.647010  710.312073
    7   Effective Speed  188.789596          NA   58.406150  700.352885
    
## Dali

    [   1][  10] Batch Time (avg: 0.1858, sd: 0.0000) Speed (avg: 688.8480) Data (avg: 0.0290, sd: 0.0498)
    [   1][  20] Batch Time (avg: 0.2215, sd: 0.0174) Speed (avg: 577.7569) Data (avg: 0.0410, sd: 0.0449)
    [   1][  30] Batch Time (avg: 0.2219, sd: 0.0173) Speed (avg: 576.8454) Data (avg: 0.0316, sd: 0.0392)
    [   1][  40] Batch Time (avg: 0.2201, sd: 0.0168) Speed (avg: 581.5356) Data (avg: 0.0243, sd: 0.0362)
    [   1][  50] Batch Time (avg: 0.2218, sd: 0.0182) Speed (avg: 577.1092) Data (avg: 0.0195, sd: 0.0337)
    [   1][  60] Batch Time (avg: 0.2217, sd: 0.0179) Speed (avg: 577.2836) Data (avg: 0.0192, sd: 0.0335)
    [   1][  70] Batch Time (avg: 0.2221, sd: 0.0174) Speed (avg: 576.3679) Data (avg: 0.0269, sd: 0.0434)
    [   1][  80] Batch Time (avg: 0.2208, sd: 0.0172) Speed (avg: 579.6026) Data (avg: 0.0258, sd: 0.0410)
    [   1][  90] Batch Time (avg: 0.2206, sd: 0.0178) Speed (avg: 580.1408) Data (avg: 0.0246, sd: 0.0393)
    [   1][ 100] Batch Time (avg: 0.2199, sd: 0.0173) Speed (avg: 582.0743) Data (avg: 0.0235, sd: 0.0377)
                 Metric     Average  Deviation         Min         Max
    0  CPU Data loading    0.000000          0         inf        -inf
    1  GPU Data Loading    0.000000          0         inf        -inf
    2  Waiting for data    0.023477  0.0377016    0.000086    0.200952
    3  CPU Compute Time    0.219903  0.0173298    0.185817    0.269321
    4  GPU Compute Time    0.217673   0.018413    0.181263    0.269239
    5   Full Batch Time    0.242848  0.0418703    0.192726    0.435221
    6     Compute Speed  582.074269         NA  475.269771  688.848003
    7   Effective Speed  527.078721         NA  294.103162  664.154863