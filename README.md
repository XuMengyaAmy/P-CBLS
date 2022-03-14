# P-CBLS: Confidence-Aware Paced-Curriculum Learning by Label Smoothing 
This repository contains the code for the paper "Confidence-Aware Paced-Curriculum Learning by Label Smoothing". 
We evaluate our CBLS and P-CBLS on three common computer vision classification datasets: Tiny-ImageNet, CIFAR10, CIFAR100, and four surgical datasets includeing multi-class classification, multi-label classification, semantic segmentation, image captioning tasks.

The datasets can be found here:

Workflow classificaton: M2CAI 2016 Challenge (http://camma.u-strasbg.fr/m2cai2016/index.php/program-challenge/)

Tool Cassification: Instrument Segmentation Challenge 2017 (https://endovissub2018-roboticscenesegmentation.grand-challenge.org/Data/) and 2018 (https://endovissub2017-roboticinstrumentsegmentation.grand-challenge.org/)

CIFAR-100 and CIFAR-100 Dataset (https://www.cs.toronto.edu/~kriz/cifar.html) 

Tiny-ImageNet Dataset (https://www.kaggle.com/c/tiny-imagenet)

The detailed README files for each task can be found inside the corresponding code folder.

## Here, we take the training commands on Tiny-ImageNet as the example to demonstrate the usage of our approaches.
## Command to train models with CBLS approach on  Tiny-ImageNet:
```
cd cv_classification
```
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
