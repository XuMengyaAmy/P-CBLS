Commands to run P-CBLS code for segmentation dataset:

1. Training command for baseline
```
python3.6 train.py --method baseline
```
2. Training command for LS
```
python3.6 train.py --method cbls --label_smoothing 0.1 --factor 1.0
```
3. Training command for CBLS (ULS)
```
python3 train.py --method cbls --label_smoothing 0.6 --factor 0.9
```

4. Training command for CBLS (SVLS)
```
python3 train.py --method cbsvls --sigma 0.9 --sigma_factor 0.5 --ksize 3
```
5. Training command for CBLS (ULS+SVLS)
```
python3 train.py --method cbls_cbsvls --label_smoothing 0.09 --factor 0.5 --sigma 0.9 --sigma_factor 0.5 --ksize 3
```
6. To obtain the optimal temperature (opt_T) based on the saved baseline checkpoint for paced learning
```
python3.6 temperature_scaling_segmentation.py
```
you will obtain the "opt_t"

7. Training command for pixel-wise P-CBLS (ULS)
```
python3.6 train.py --method p-cbls --opt_t xx --label_smoothing 0.6 --factor 0.9 --lamda 0.9 --E_all 0.4
```
8. Training command for pixel-wise P-CBLS (SVLS)
```
python3.6 train.py --method p-cbsvls --opt_t xx --sigma 0.9 --sigma_factor 0.5 --ksize 3 --lamda 0.8 --E_all 0.4
```

