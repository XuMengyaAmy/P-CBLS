# CBLS
## Command to train different models:
CE_LS_CBLS:
```
CUDA_VISIBLE_DEVICES=0 python main.py --mode CE --num_epochs 200 --dataset cifar10
```
```
CUDA_VISIBLE_DEVICES=0 python main.py --mode LS --ls_factor 0.1 --num_epochs 200 --dataset cifar10
```

```
CUDA_VISIBLE_DEVICES=0 python main.py --mode CBLS --ls_factor 0.4 --use_cls --cls_decay 0.94 --num_epochs 200 --dataset cifar10
```

```
CUDA_VISIBLE_DEVICES=1 python main.py --mode CBLS --ls_factor 0.4 --use_cls --cls_decay 0.90 --num_epochs 200 --dataset cifar10
```

OLS:
```
CUDA_VISIBLE_DEVICES=0 python main.py --mode OLS --num_epochs 200 --dataset cifar10
```
```
CUDA_VISIBLE_DEVICES=1 python main.py --mode OLS --num_epochs 200 --dataset cifar100
```

DL:
```
CUDA_VISIBLE_DEVICES=0 python main.py --mode OLS --num_epochs 200 --dataset cifar10
```

```
CUDA_VISIBLE_DEVICES=1 python main.py --mode OLS --num_epochs 200 --dataset cifar100
```
Ablation Study on CIFAR100:
```
mode:CLS:0.1  epoch: 99  acc: 66.33  best epoch: 92  best acc: 66.47, ls:0.00, lr:0.0001
mode:CLS:0.2  epoch: 99  acc: 66.19  best epoch: 90  best acc: 66.26, ls:0.00, lr:0.0001
mode:CLS:0.3  epoch: 99  acc: 66.34  best epoch: 87  best acc: 66.49, ls:0.00, lr:0.0001
mode:CLS:0.2 epoch: 149  acc: 66.23  best epoch: 106  best acc: 66.35, ls:0.00, lr:0.0001
mode:CLS:0.3 epoch: 149  acc: 66.35  best epoch: 148  best acc: 66.51, ls:0.00, lr:0.0001
mode:CLS:0.4 epoch: 149  acc: 66.45  best epoch: 141  best acc: 66.48, ls:0.00, lr:0.0001
mode:CLS:0.5 epoch: 149  acc: 65.93  best epoch: 79  best acc: 66.08, ls:0.00, lr:0.0001
mode:CLS:0.6 epoch: 149  acc: 66.04  best epoch: 94  best acc: 66.32, ls:0.00, lr:0.0001
mode:CLS:0.7 epoch: 149  acc: 65.84  best epoch: 142  best acc: 66.08, ls:0.00, lr:0.0001
mode:CLS:0.8 epoch: 149  acc: 64.71  best epoch: 96  best acc: 64.91, ls:0.00, lr:0.0001

mode:CLS:0.4 decay:0.90 epoch: 149  acc: 66.45  best epoch: 141  best acc: 66.48, ls:0.00, lr:0.0001
mode:CLS:0.4 decay:0.92 epoch: 199  acc: 66.25  best epoch: 195  best acc: 66.45, ls:0.00, lr:0.0001
mode:CLS:0.4 decay:0.93 epoch: 199  acc: 66.46  best epoch: 169  best acc: 66.57, ls:0.00, lr:0.0001
mode:CLS:0.4 decay:0.94 epoch: 199  acc: 67.18  best epoch: 185  best acc: 67.26, ls:0.00, lr:0.0001
mode:CLS:0.4 decay:0.96 epoch: 199  acc: 66.47  best epoch: 178  best acc: 66.58, ls:0.00, lr:0.0001
mode:CLS:0.4 decay:0.95 epoch: 199  acc: 66.98  best epoch: 173  best acc: 67.17, ls:0.00, lr:0.0001
```
Ablation Study on CIFAR10:
```
mode:CE:0.0 decay:0.9 epoch: 199  acc: 83.63  best epoch: 63  best acc: 84.14, ls:0.00, lr:0.0001
mode:LS:0.1 decay:0.9 epoch: 199  acc: 84.43  best epoch: 75  best acc: 84.70, ls:0.10, lr:0.0001
mode:CLS:0.4 decay:0.94 epoch: 199  acc: 88.35  best epoch: 124  best acc: 88.59, ls:0.00, lr:0.0001
mode:CLS:0.4 decay:0.9 epoch: 199  acc: 87.35  best epoch: 63  best acc: 87.64, ls:0.00, lr:0.0001
```
