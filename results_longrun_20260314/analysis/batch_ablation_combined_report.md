# Batch Ablation Combined Report (4 Optimizers)

## Overall Batch Size Recommendation

- Batch size ổn nhất (overall): **32** (scaled_lr=0.00125, acc=95.563±2.821, loss=0.1445±0.1248)
- Batch size kém nhất (overall): **512** (scaled_lr=0.02, acc=94.699±2.416, loss=0.1748±0.1071)

## Best per Optimizer

- Adam: batch=32, lr=0.00125, acc=97.433±0.093, loss=0.0582±0.0008
- AdamW: batch=32, lr=0.00125, acc=97.480±0.321, loss=0.0536±0.0010
- SGD: batch=32, lr=0.00125, acc=90.898±0.246, loss=0.3491±0.0079
- SGD_Momentum: batch=256, lr=0.01, acc=96.457±0.053, loss=0.1181±0.0018

## Learning Rate Mapping (Linear Scaling in this ablation)

- batch=32 -> lr=0.00125
- batch=256 -> lr=0.01
- batch=512 -> lr=0.02