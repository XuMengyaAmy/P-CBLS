import pickle
import torch
import numpy as np
import random
import os

seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)


def as_num(x):
    y = '{:.10f}'.format(x) # .10f means keep 10 decimal
    return y

# To prepare the dataset for curriculum learning
## Step 1: to get the image_path_label_confidence file
image_path_label = []
# file_list = 'data/SPL_data/train.lst'
file_list = 'data/sample_wise_data/train.lst'
#read txt method one
f = open(file_list)
line = f.readline().strip('\n')
while line:
    # print(line)
    image_path_label.append(line)
    line = f.readline().strip('\n')
f.close()


confidence = []
file_confidence ='data/sample_wise_data/LinkNet34/ts_confidence_score.txt'
# print('file_confidence', file_confidence)

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

image_path_confidence = []
image_path_confidence = ['%s\t%s' % (image_path_label[i], confidence[i])\
                    for i in range(len(image_path_label))]
# print('image_path_confidence', image_path_confidence)


with open(os.path.join('data/sample_wise_data/LinkNet34/', 'image_path_ts_confidence.lst'), 'w') as f_writer:        
    f_writer.write('\n'.join(image_path_confidence))
    f_writer.write('\n') # add the empty line in the end to solve the sort issue. (Super inportant)

## Step 2: to get the sorted image_path_label_confidence file
# https://blog.csdn.net/v1_vivian/article/details/74980074
# sort the content based on the confidence score

sorted = ''.join(sorted(open(os.path.join('data/sample_wise_data/LinkNet34/', 'image_path_ts_confidence.lst')), key=lambda s: s.split()[1], reverse=1))
print('sorted', sorted)


with open(os.path.join('data/sample_wise_data/LinkNet34/', 'sorted_image_path_ts_confidence.lst'), 'w') as f_writer:
    f_writer.write(sorted)
print('Finish the sorting')



