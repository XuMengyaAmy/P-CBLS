
import random
from data import ImageDetectionsField, TextField, RawField
from data import COCO, DataLoader
import evaluation
from evaluation import PTBTokenizer, Cider

# for non-DANN
from models.transformer import Transformer, MemoryAugmentedEncoder, MeshedDecoder, ScaledDotProductAttentionMemory

# # for DANN
# from models.transformer import MemoryAugmentedEncoder, MeshedDecoder, ScaledDotProductAttentionMemory # Transformer, 
# from models.transformer.transformer_DANN  import Transformer


import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import NLLLoss
from tqdm import tqdm
import argparse, os, pickle
import numpy as np
import itertools
import multiprocessing
from shutil import copyfile
import warnings
warnings.filterwarnings("ignore")
import os, json


seed = 1234
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def evaluate_metrics(model, dataloader, text_field):
    import itertools
    model.eval()
    gen = {}
    gts = {}
    with tqdm(desc='evaluation', unit='it', total=len(dataloader)) as pbar:
        for it, (images, caps_gt) in enumerate(iter(dataloader)):
            images = images.to(device)
            with torch.no_grad():
                out, _ = model.beam_search(images, 20, text_field.vocab.stoi['<eos>'], 5, out_size=1)  
            caps_gen = text_field.decode(out, join_words=False)
            for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
                gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                gen['%d_%d' % (it, i)] = [gen_i, ]    
                gts['%d_%d' % (it, i)] = gts_i
            pbar.update()


    if not os.path.exists('predict_caption'):
        os.makedirs('predict_caption')
    json.dump(gen, open('predict_caption/predict_caption_val.json', 'w'))

    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)
    scores, _ = evaluation.compute_scores(gts, gen)
    return scores



if __name__ == '__main__':
    device = torch.device('cuda')
    parser = argparse.ArgumentParser(description='Incremental domain adaptation for surgical report generation')
    parser.add_argument('--exp_name', type=str, default='m2_transformer')
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--m', type=int, default=40)   
    parser.add_argument('--head', type=int, default=8)
    parser.add_argument('--warmup', type=int, default=10000)
    parser.add_argument('--features_path', type=str)
    parser.add_argument('--features_path_DA', type=str)
    parser.add_argument('--annotation_folder', type=str)
    parser.add_argument('--annotation_folder_DA', type=str)

    args = parser.parse_args()
    print(args)
   
    print('Validation')

    # Pipeline for image regions
    image_field = ImageDetectionsField(detections_path=args.features_path, max_detections=6, load_in_tmp=False)  
    # Pipeline for text
    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy', remove_punctuation=True, nopoints=False)

    # Create the dataset
    dataset = COCO(image_field, text_field, args.features_path, args.annotation_folder, args.annotation_folder)
    train_dataset, val_dataset = dataset.splits   
    print('train:', len(train_dataset))
    print('val:', len(val_dataset))
    

    if not os.path.isfile('vocab_%s.pkl' % args.exp_name):
        print("Building vocabulary")
        text_field.build_vocab(train_dataset, val_dataset, min_freq=2)  
        pickle.dump(text_field.vocab, open('vocab_%s.pkl' % args.exp_name, 'wb'))
    else:
        text_field.vocab = pickle.load(open('vocab_%s.pkl' % args.exp_name, 'rb'))

    print('vocabulary size is:', len(text_field.vocab))
    print(text_field.vocab.stoi)

    
    encoder = MemoryAugmentedEncoder(3, 0, attention_module=ScaledDotProductAttentionMemory, attention_module_kwargs={'m': args.m}) 
    decoder = MeshedDecoder(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>'])
    model = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)
    # print("model", model)

    dict_dataset_val = val_dataset.image_dictionary({'image': image_field, 'text': RawField()})

    data = torch.load('/saved_models/CBLS/cbls_ls_0.1/%s_best.pth' % args.exp_name)
    model.load_state_dict(data['state_dict'])
    print("Epoch %d" % data['epoch'])  
    print(data['best_cider'])

    #dataloader_train = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
    
    # To calculate the confidence in order for train dataset, set the shuffle=False in dataloader_train is super important and drop_last=False is also important
    dataloader_train = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, drop_last=False)
    
    dict_dataloader_val = DataLoader(dict_dataset_val, batch_size=args.batch_size)

    # scores = evaluate_metrics(model, dict_dataloader_val, text_field)  
    # print("Validation scores :", scores)

import torch.nn.functional as F
def calculate_class_conf(model, dataloader):
    model.eval()
    true_class_wise_conf = []
    with torch.no_grad():
        with tqdm(desc='calculate confidence', unit='it', total=len(dataloader)) as pbar:
            for it, (detections, captions) in enumerate(dataloader):
                detections, captions = detections.to(device), captions.to(device)
                out = model(detections, captions)
                captions_gt = captions[:, 1:].contiguous() # torch.Size([50, 15])
                out = out[:, :-1].contiguous() # torch.Size([50, 15, 41])
                out = F.softmax(out, dim=1) # output = F.softmax(out, dim=1)
                # print(captions_gt.shape[0]) # 50
                for idx in range(captions_gt.shape[0]): # captions_gt[0] is the batch_size, represents num_samples
                    true_class_probability_sum_one_sample = 0
                    output = out[idx]
                    # print('output for one sample', output.shape) # torch.Size([15, 41]) 15 wors in predicted sentece, 41 probability for the 41 classes in dictionary

                    caption_gt = captions_gt[idx]
                    # print('caption_gt for one sample', caption_gt) # tensor([10,  6, 12, 14,  6, 12,  7,  8,  9, 13, 16,  3,  1,  1,  1], # 15 gt words, represents
                    # print('caption_gt shape for one sample', caption_gt.shape) # torch.Size([15])
                    # print(caption_gt.shape[0]) # 15
                    
                    for idx in range(caption_gt.shape[0]): # 15 prediction words
                        true_class_probability_sum_one_sample += output[idx, caption_gt[idx]].item()
                    
                    confidence = true_class_probability_sum_one_sample / caption_gt.shape[0] # average them
                    # print('confidence', confidence)
                    true_class_wise_conf.append(confidence)
                pbar.update()
    return true_class_wise_conf

true_class_wise_conf = calculate_class_conf(model, dataloader_train)
print('Sample size:', len(true_class_wise_conf)) # Sample size: 1550, because of the drop_last. To set it False

def as_num(x):
    y = '{:.10f}'.format(x) # .10f means keep 10 decimal
    return y 


filename = open('IDA_SurgicalReport/self_paced_learning/confidence_score.txt','w')
for i in true_class_wise_conf:
    # =========================================== #
    # To solve the issue of "9.997991e-08"
    if ('E' in str(i) or 'e' in str(i)):
        i = as_num(float(str(i)))
    # =========================================== #
    filename.write(str(i))
    filename.write('\n')
filename.close()
print('Done')



