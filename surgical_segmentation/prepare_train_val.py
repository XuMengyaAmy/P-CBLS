from prepare_data import data_path




def get_split(fold):
    train_path = data_path / '2018_original' / 'train'
    val_path = data_path / '2018_original' / 'val'

    train_file_names = []
    val_file_names = []

    train_file_names = list((train_path / 'images').glob('*')) 
    val_file_names = list((val_path / 'images').glob('*'))

    return train_file_names, val_file_names


def get_split_resized(fold):
    train_path = data_path / '2018_down' / 'train'
    val_path = data_path / '2018_down' / 'val'

    train_file_names = []
    val_file_names = []

    train_file_names = list((train_path / 'images').glob('*'))
    val_file_names = list((val_path / 'images').glob('*'))

    return train_file_names, val_file_names

# a, b = get_split(0)
# print("Train:", '\n', a[-1]) # data/2018_original/train/images/seq_10_frame004.png
# print("Val:", '\n', b[-1]) # data/2018_original/val/images/seq_2_frame089.png


###### not in sequence order, when to get the confidence score, we need the sequence ######
import os
import pathlib

def get_split_order(fold):
    train_path = data_path / '2018_original' / 'train'
    val_path = data_path / '2018_original' / 'val'

    train_file_names = []
    val_file_names = []

    train_file_names = list((train_path / 'images').glob('*')) 
    val_file_names = list((val_path / 'images').glob('*'))

    # Train
    train_file_names_seq1to9 = []
    train_file_names_seq10to16 = []
    train_file_names_order = []
    for i in range(len(train_file_names)):
        if int(str(train_file_names[i]).split("/")[-1].split(".")[0].split("_")[1]) < 10:
            train_file_names_seq1to9.append(train_file_names[i])
        else:
            train_file_names_seq10to16.append(train_file_names[i])
    
    train_file_names_seq1to9.sort()
    train_file_names_seq10to16.sort()

    train_file_names_order += train_file_names_seq1to9
    train_file_names_order += train_file_names_seq10to16
    print('Train size:', len(train_file_names_order))

    
    # Val
    val_file_names_seq1to9 = []
    val_file_names_seq10to16 = []
    val_file_names_order = []
    for i in range(len(val_file_names)):

        if int(str(val_file_names[i]).split("/")[-1].split(".")[0].split("_")[1]) < 10:
            val_file_names_seq1to9.append(val_file_names[i])
        else:
            val_file_names_seq10to16.append(val_file_names[i])
    
    val_file_names_seq1to9.sort()
    val_file_names_seq10to16.sort()

    val_file_names_order += val_file_names_seq1to9
    val_file_names_order += val_file_names_seq10to16
    print('Val size:', len(val_file_names_order))

    

    ordered_file_path = 'data/SPL_data' # SPL: self-paced learning
    if not os.path.isdir(ordered_file_path):
        pathlib.Path(ordered_file_path).mkdir(parents=True, exist_ok=True)
    
    str_train_file_names_order = [str(i) for i in train_file_names_order]  
    str_val_file_names_order = [str(i) for i in val_file_names_order]
    print('str_train_file_names_order Size:', len(str_train_file_names_order))
    print('str_val_file_names_order Size:', len(str_val_file_names_order))
    with open(os.path.join(ordered_file_path, 'train.lst'), 'w') as f_writer:
        f_writer.write('\n'.join(str_train_file_names_order))
    with open(os.path.join(ordered_file_path, 'valid.lst'), 'w') as f_writer:
        f_writer.write('\n'.join(str_val_file_names_order))

    return train_file_names_order, val_file_names_order



def get_split_SPL(fold, train_file_spl, easy_ratio = 0.8):
    easy_train_file_names = []
    with open(train_file_spl) as file:
        line = file.readlines()
        for i, rows in enumerate(line):
            if i in range(0, int(len(line)*easy_ratio)):
                easy_train_file_names.append(pathlib.Path(rows.split("\t")[0]))
    
    file.close()
    return easy_train_file_names
