import os
import sys
import random
import numpy as np
from glob import glob
from torch.utils.data import Dataset, DataLoader
import json

import h5py
import re

'''
To sort the training samples based on the confidence score.txt
'''

# [{"id_path": "seq_2/roi_features_resnet18_inc_sup_cbs/frame014_node_features.npy", \
# "caption": "bipolar forceps and large needle driver are looping tissue, prograsp forceps is manipulating tissue"},


# Original annotation file
ann_root = 'annotations_new/annotations_miccai_inc_sup_cbs'
dataset = json.load(open(os.path.join(ann_root, 'captions_train.json'), 'r'))
# print('dataset', dataset)


# Store all confidence value into a list
confidence = []
file_confidence = os.path.join('/IDA_SurgicalReport/self_paced_learning/confidence_score.txt')
f = open(file_confidence)
line = f.readline().strip('\n')
while line:
    # print(line)
    # =========================================== #
    # To solve the issue of "9.997991e-08 > 0.9"
    if ('E' in line or 'e' in line):
        line = as_num(float(line))
    # =========================================== #
    confidence.append(line)
    line = f.readline().strip('\n')
f.close()


# Add confidence into the annotation file

for i in range(len(dataset)): # i is dictionary
    dataset[i]['confidence'] = confidence[i]

# print('dataset', dataset)

# Code Reference: https://python3-cookbook.readthedocs.io/zh_CN/latest/c01/p13_sort_list_of_dicts_by_key.html
from operator import itemgetter
sorted_dataset_by_confidence = sorted(dataset, key=itemgetter('confidence'), reverse= True)
# print('sorted_dataset_by_confidence', sorted_dataset_by_confidence)

if not os.path.exists('/IDA_SurgicalReport/self_paced_learning/'):
    os.makedirs('/IDA_SurgicalReport/self_paced_learning/')

with open('/IDA_SurgicalReport/self_paced_learning/sorted_captions_train.json', 'w') as f:
    json.dump(sorted_dataset_by_confidence, f)

print('Done!')
