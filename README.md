# P-CBLS: Confidence-Aware Paced-Curriculum Learning by Label Smoothing 
This repository contains the code for the paper "Confidence-Aware Paced-Curriculum Learning by Label Smoothing". 
We evaluate our CBLS and P-CBLS on three common computer vision classification datasets: Tiny-ImageNet, CIFAR10, CIFAR100, and four surgical datasets includeing multi-class classification, multi-label classification, semantic segmentation, image captioning tasks.

Here, we take the the training commands on Tiny-ImageNet as the example to demonstrate the usage of our appraoches.

## Command to train models with CBLS approach on  Tiny-ImageNet:
Our CBLS:

```
python main.py --mode CBLS --ls_factor 0.38 --use_cls --cls_decay 0.5 --num_epochs 200 --dataset tinyimagenet
```

Other baselines including Cross-Entropy Loss (CE), Label Smoothing (LS), Online Label Smoothing (OLS) and Disturb Label (DL).
CE
```
python main.py --mode CE --num_epochs 200 --dataset tinyimagenet
```

LS
```
python main.py --mode LS --ls_factor 0.1 --num_epochs 200 --dataset tinyimagenet
```

OLS:
```
python main.py --mode OLS --num_epochs 200 --dataset tinyimagenet
```

DL:
```
python main.py --mode DL --num_epochs 200 --dataset tinyimagenet
```

## Command to train models with P-CBLS approach on  Tiny-ImageNet:
Obtain the optimal temperature value based on the CE baseline.
```
python test_temperature.py
```

Obtain the sorted samples sorted by the calibrated confidence value after pass the optimal temperature value
```
python cal_confidence.py
```
Our P-CBLS
```
python main_pcbls.py
```
