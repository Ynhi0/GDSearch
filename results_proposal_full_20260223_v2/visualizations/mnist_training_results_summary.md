## MNIST Training Results - Tabular Summary

| Optimizer/Config | Final Loss | Loss Std | Final Test Acc (%) | Acc Std (%) | Speed (iters/sec) |
|------------------|------------|----------|--------------------|-------------|-------------------|
| training_results_summary | 0.3050 | N/A | 100.0000 | 0.0000 | N/A |
| SimpleMLP_GradientClip_5.0 | 0.0128 | 0.0003 | 98.1120 | 0.0685 | N/A |
| SimpleMLP_GradientClip_None | 0.0128 | 0.0003 | 98.1120 | 0.0685 | N/A |
| SimpleMLP_GradientClip_10.0 | 0.0128 | 0.0003 | 98.1120 | 0.0685 | N/A |
| SimpleMLP_GradientClip_1.0 | 0.0129 | 0.0004 | 98.0910 | 0.0639 | N/A |
| SimpleMLP_GradientClip_0.5 | 0.0218 | 0.0007 | 98.0350 | 0.0639 | N/A |
| batch_ablation_seeds42_123_456_789_1011_1213_1415_1617_1819_2021 | 0.1169 | N/A | 96.4700 | 0.0000 | N/A |
| Baseline | 0.2125 | 0.0095 | 94.3240 | 0.2730 | N/A |
| SimpleMLP_SAM_rho_0.2 | 0.2331 | 0.0017 | 93.9030 | 0.0806 | N/A |
| SimpleMLP_SAM_rho_0.1 | 0.2367 | 0.0020 | 93.6870 | 0.0725 | N/A |
| SimpleMLP_SAM_rho_0.05 | 0.2414 | 0.0022 | 93.5240 | 0.0687 | N/A |
| SimpleMLP_SAM_rho_0.02 | 0.2448 | 0.0023 | 93.3880 | 0.0894 | N/A |
| SimpleMLP_SAM_rho_0.01 | 0.2458 | 0.0023 | 93.3120 | 0.1085 | N/A |
| SchedulerAblation_MNIST_SGD | 0.3050 | N/A | 1.0000 | 0.0000 | N/A |
| BatchAblation_MNIST_SAM | 0.1190 | 0.0022 | N/A | N/A | N/A |
| BatchAblation_MNIST_SGD | 0.1247 | 0.0013 | N/A | N/A | N/A |
