import pickle
import os
import json
import pandas as pd
import torch
import numpy as np
import random
import argparse

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../data/Train_risk_classification_ans.csv')
    args = parser.parse_args()

    return args

def same_seeds(seed):
	  torch.manual_seed(seed)
	  if torch.cuda.is_available():
            torch.cuda.set_device(2)
	  np.random.seed(seed)
	  random.seed(seed)
	  torch.backends.cudnn.benchmark = False
	  torch.backends.cudnn.deterministic = True
same_seeds(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()

class RISK_preprocess:

    def __init__(self, filename):

        self.filename = filename
    
    def process(self):

        # load file
        df = pd.read_csv(self.filename, sep=',')

        if 'Train' not in self.filename:

            dev_text = []
            for i in df['text']:
                dev_text.append(i.replace('個管師','').replace('民眾','').replace('：',''))

            df = pd.DataFrame({'id':df['article_id'],'text':dev_text,'label':df['label']})

            filename = self.filename[:self.filename.find('_')]
            return df.to_csv(f'{filename}_risk_preprocess.csv', index=None)

        else:

            df['label'][df['label']=='１']='1'
            df['label'][df['label']=='０']='0'
            
            possible_labels = df.label.unique()

            label_dict = {}
            for index, possible_label in enumerate(possible_labels):
                label_dict[possible_label] = index
            df['new_label'] = df.label.replace(label_dict)
            
            train_text = []
            for i in df['text']:
                train_text.append(i.replace('個管師','').replace('民眾','').replace('：',''))
    
            df = pd.DataFrame({'id':df['article_id'],'text':train_text,'label':df['new_label']})

            filename = self.filename[:self.filename.find('_')]
            return df.to_csv(f'{filename}_risk_preprocess.csv', index=None)

if __name__ == '__main__':

    args = parse_args()
    args = vars(args)
    preprocess = RISK_preprocess(args["data_dir"])
    df = preprocess.process()

    # python3 RISK_preprocess.py --data_dir ../data/Train_risk_classification_ans.csv
    # python3 RISK_preprocess.py --data_dir ../data/Develop_risk_classification.csv
    # python3 RISK_preprocess.py --data_dir ../data/Test_risk_classification.csv