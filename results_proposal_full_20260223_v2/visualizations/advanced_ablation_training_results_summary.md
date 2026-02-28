## Advanced Ablation Training Results - Tabular Summary

| Optimizer/Config | Final Loss | Loss Std | Final Test Acc (%) | Acc Std (%) | Speed (iters/sec) |
|------------------|------------|----------|--------------------|-------------|-------------------|
| LabelSmoothing_EMA | 0.7314 | 0.0049 | 95.2530 | 0.1705 | N/A |
| LabelSmoothing_only | 0.7325 | 0.0048 | 95.2190 | 0.1849 | N/A |
| Baseline | 0.2125 | 0.0095 | 94.3240 | 0.2730 | N/A |
| EMA_only | 0.2131 | 0.0108 | 94.2610 | 0.3087 | N/A |
| AMP_LabelSmoothing | 0.7843 | 0.0105 | 94.0310 | 0.3528 | N/A |
| All_Combined | 0.7840 | 0.0104 | 93.9900 | 0.3446 | N/A |
| AMP_EMA | 0.2750 | 0.0204 | 92.7380 | 0.4701 | N/A |
| AMP_only | 0.2755 | 0.0203 | 92.7380 | 0.5092 | N/A |
