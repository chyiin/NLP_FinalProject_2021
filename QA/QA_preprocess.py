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
    parser.add_argument('--data_dir', type=str, default='../data/Train_qa_ans.json')
    args = parser.parse_args()

    return args

def same_seeds(seed):
	  torch.manual_seed(seed)
	  if torch.cuda.is_available():
            torch.cuda.set_device(0)
	  np.random.seed(seed)
	  random.seed(seed)
	  torch.backends.cudnn.benchmark = False
	  torch.backends.cudnn.deterministic = True
same_seeds(123)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()

class QA_preprocess:

    def __init__(self, filename):

        self.filename = filename
    
    def process(self):

        # load file
        with open(os.path.join('./', self.filename)) as f:
            data = json.load(f)

        # keep in list
        sample_id = []
        sample_text = []
        sample_question = []
        sample_answer = []
        sample_choice = []

        for i in data:
            sample_id.append(i['id'])
            sample_text.append(i['text'])
            sample_question.append(i['question']['stem'])
            if 'Train' in self.filename: 
                if (i['answer'] == 'C ') | (i['answer'] == 'Ｃ'):
                    sample_answer.append('C')
                elif (i['answer'] == 'A ') | (i['answer'] == 'Ａ'):
                    sample_answer.append('A')
                elif (i['answer'] == 'B ') | (i['answer'] == 'Ｂ'):
                    sample_answer.append('B')
                else:
                    sample_answer.append(i['answer'])
            c = []
            for j in i['question']['choices']:
                c.append(j['text'])
            sample_choice.append(c)

        if 'Train' not in self.filename:

            df = pd.DataFrame({'id':sample_id,'text':sample_text,'question':sample_question,'choice':sample_choice})

            dev_text = [] 

            for i in range(len(df['text'])):
                splt = [x+'。' for x in df['text'][i].split('。')]
                tx = []
                for j in range(len(splt)):
                    n = 0
                    for g in splt[j]:
                        if g in list(df['question'][i]):
                            n = n + 1
                    if n > 5:
                        tx.append(''.join(splt[j:j+3]))
                if len(''.join(tx)) == 0:
                    dev_text.append(df['text'][i])
                else:
                    dev_text.append(''.join(tx))

            cho_dev = []
            for i in range(len(dev_text)):
                tx = []
                for j in df['choice'][i]:
                    tx.append('[SEP]'+j)
                cho_dev.append(''.join(tx))

            df['qa'] = df['question'] + cho_dev +  ['[SEP]']*df.shape[0] + dev_text
            
            filename = self.filename[:self.filename.find('_')]
            return df.to_csv(f'{filename}_qa_preprocess.csv', index=None)

        else:

            df = pd.DataFrame({'id':sample_id,'text':sample_text,'question':sample_question,'choice':sample_choice,'answer':sample_answer})
            df = df[df['answer']!='菜花'].reset_index()

            possible_labels = ['A','B','C']

            label_dict = {}
            for index, possible_label in enumerate(possible_labels):
                label_dict[possible_label] = index

            df['label'] = df.answer.replace(label_dict)

            train_text = [] 

            for i in range(len(df['text'])):
                splt = [x+'。' for x in df['text'][i].split('。')]
                tx = []
                for j in range(len(splt)):
                    n = 0
                    for g in splt[j]:
                        if g in list(df['question'][i]):
                            n = n + 1
                    if n > 5:
                        tx.append(''.join(splt[j:j+3]))
                if len(''.join(tx)) == 0:
                    train_text.append(df['text'][i])
                else:
                    train_text.append(''.join(tx))

            cho = []
            for i in range(len(train_text)):
                tx = []
                for j in df['choice'][i]:
                    tx.append('[SEP]'+j)
                cho.append(''.join(tx))

            df['qa'] = df['question'] + cho + ['[SEP]']*df.shape[0] + train_text
            df = df.sample(frac=1).reset_index()
            
            filename = self.filename[:self.filename.find('_')]
            return df.to_csv(f'{filename}_qa_preprocess.csv', index=None)

if __name__ == '__main__':

    args = parse_args()
    args = vars(args)
    preprocess = QA_preprocess(args["data_dir"])
    df = preprocess.process()

    # python3 QA_preprocess.py --data_dir ../data/Train_qa_ans.json
    # python3 QA_preprocess.py --data_dir ../data/Develop_QA.json
    # python3 QA_preprocess.py --data_dir ../data/Test_QA.json