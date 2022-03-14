Data preparation
To run the P-CBLS code on captioning dataset, annotations folder and features folder for the dataset are needed. 
You can download them from other work.

1. Training command for baseline
```
python3.8 train_CBLS.py --features_path xxxxx --annotation_folder xxxxx
```
2. Training command for LS
```
python3.8 train_CBLS.py --features_path xxx --annotation_folder xxxx --cbls True --cbls_constant True --label_smoothing 0.1
```
3. Training command for CBLS
```
python3.8 train_CBLS.py --features_path xxx --annotation_folder xxx --cbls True --cbls_decrease True --label_smoothing xx --factor xx
```

4. Command for sorting samples for pacing function
```
python3.8 val_calculate_task_difficulty_with_confidence.py
```

5. Training command for P-CBLS
```
python3.8 train_CBLS_linearMPL.py --features_path xxx --annotation_folder xxx --cbls True --cbls_decrease True  --label_smoothing xx --factor xx --initial_sample xx --all_sample_epoch_ratio xx
```

