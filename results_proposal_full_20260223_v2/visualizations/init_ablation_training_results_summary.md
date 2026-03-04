## Init Ablation Training Results - Tabular Summary

| Optimizer/Config | Final Loss | Loss Std | Final Test Acc (%) | Acc Std (%) | Speed (iters/sec) |
|------------------|------------|----------|--------------------|-------------|-------------------|
| InitAblation_kaiming_uniform_SGD_Momentum | 0.1264 | 0.0056 | 96.1880 | 0.5639 | N/A |
| InitAblation_kaiming_normal_SGD_Momentum | 0.1278 | 0.0063 | 95.9490 | 0.6352 | N/A |
| InitAblation_kaiming_normal_Adam | 0.1329 | 0.0087 | 95.9080 | 0.4614 | N/A |
| InitAblation_kaiming_uniform_Adam | 0.1345 | 0.0079 | 95.8530 | 0.4954 | N/A |
| InitAblation_kaiming_uniform_AdamW | 0.1336 | 0.0070 | 95.7860 | 0.4207 | N/A |
| InitAblation_xavier_uniform_SGD_Momentum | 0.1479 | 0.0048 | 95.7730 | 0.4478 | N/A |
| InitAblation_kaiming_normal_AdamW | 0.1335 | 0.0083 | 95.7700 | 0.6598 | N/A |
| InitAblation_xavier_normal_AdamW | 0.1588 | 0.0121 | 95.6340 | 0.4492 | N/A |
| InitAblation_xavier_normal_SGD_Momentum | 0.1494 | 0.0050 | 95.6030 | 0.5107 | N/A |
| InitAblation_xavier_normal_Adam | 0.1569 | 0.0105 | 95.5400 | 0.5455 | N/A |
| InitAblation_xavier_uniform_Adam | 0.1609 | 0.0078 | 95.3320 | 0.4220 | N/A |
| InitAblation_xavier_uniform_AdamW | 0.1601 | 0.0071 | 95.2620 | 0.5332 | N/A |
| InitAblation_uniform_small_SGD_Momentum | 0.1757 | 0.0038 | 95.2490 | 0.4638 | N/A |
| InitAblation_uniform_small_Adam | 0.1969 | 0.0100 | 94.5120 | 0.4383 | N/A |
| InitAblation_uniform_small_AdamW | 0.1945 | 0.0069 | 94.5070 | 0.4928 | N/A |
| InitAblation_kaiming_uniform_SGD | 0.4978 | 0.0300 | 86.5710 | 1.8702 | N/A |
| InitAblation_normal_small_AdamW | 0.4769 | 0.1430 | 85.8940 | 6.1565 | N/A |
| InitAblation_normal_small_Adam | 0.4763 | 0.1393 | 85.5530 | 5.7402 | N/A |
| InitAblation_kaiming_normal_SGD | 0.5127 | 0.0298 | 85.2420 | 2.5903 | N/A |
| InitAblation_xavier_uniform_SGD | 1.1400 | 0.0253 | 64.1600 | 4.1967 | N/A |
| InitAblation_xavier_normal_SGD | 1.1274 | 0.0389 | 63.2500 | 4.5088 | N/A |
| InitAblation_uniform_small_SGD | 1.6774 | 0.0510 | 38.3860 | 1.7550 | N/A |
| InitAblation_normal_small_SGD_Momentum | 2.2734 | 0.0828 | 13.1330 | 5.3490 | N/A |
| InitAblation_normal_small_SGD | 2.3012 | 0.0000 | 11.3500 | 0.0000 | N/A |
