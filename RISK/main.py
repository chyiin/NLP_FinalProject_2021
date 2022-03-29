from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertConfig
import torch
import random
import numpy as np
from transformers import get_linear_schedule_with_warmup
import pandas as pd
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from seqeval.metrics import f1_score, accuracy_score
import torch.nn.functional as F
import argparse
from config import *

# argument settting
def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../data/Train_risk_preprocess.csv')
    parser.add_argument('--model_file', type=str, default='./risk_model.pt', help='Model file name.')
    parser.add_argument('--mode', default='train', choices=['train', 'predict'])

    args = parser.parse_args()

    return args

# set seed
def same_seeds(seed):
	  torch.manual_seed(seed)
	  if torch.cuda.is_available():
            torch.cuda.set_device(2)
	  np.random.seed(seed)
	  random.seed(seed)
	  torch.backends.cudnn.benchmark = False
	  torch.backends.cudnn.deterministic = True
same_seeds(0)

# device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()

# tokenize sentence function
def tokenize(text, MAX_LENGTH, tokenizer, label=None):

    input_ids = []
    attention_masks = []

    for sent in text:

        encoded_dict = tokenizer.encode_plus(
                            sent,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = MAX_LENGTH,           # Pad & truncate all sentences.
                            pad_to_max_length = True,
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',     # Return pytorch tensors.
                    )
        
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    if label is not None:  
        labels = torch.tensor(list(map(int, label)))
        return input_ids, attention_masks, labels
    return input_ids, attention_masks

# main function
def main():

    args = parse_args()
    args = vars(args)

    if args['mode'] == 'predict':
        evaluate(args)
    else:
        train(args)

def train(args):

    # load training data
    df = pd.read_csv(args['data_dir'],sep=',')

    # transform label types
    possible_labels = ['1','0']

    label_dict = {}
    for index, possible_label in enumerate(possible_labels):
        label_dict[possible_label] = index
    label_dict_inverse = {v: k for k, v in label_dict.items()}

    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    train_input_ids, train_attention_masks, train_labels = tokenize(df['text'], MAX_LENGTH, tokenizer, df['label'])

    # Combine the training inputs into a TensorDataset.
    dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels)

    # Create a train-validation split.
    train_size = int(DEV_SPLIT * len(dataset))
    valid_size = len(dataset) - train_size

    # Divide the dataset by randomly selecting samples.
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

    batch_size = BATCH_SIZE

    # Create the DataLoaders for our training and validation sets.
    train_dataloader = DataLoader(train_dataset, sampler = RandomSampler(train_dataset), batch_size = BATCH_SIZE)
    valid_dataloader = DataLoader(valid_dataset,sampler = SequentialSampler(valid_dataset),batch_size = BATCH_SIZE)

    # Load BertForSequenceClassification, the pretrained BERT model (bert-base-chinese)
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-chinese", 
        num_labels = LABEL_NUM, 
        output_attentions = False, 
        output_hidden_states = False, 
    ).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=BETAS, eps=EPS)     
    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEP,
        num_training_steps=len(train_dataloader)*EPOCHS
    )

    # training
    loss_values, validation_loss_values, valid_acc = [], [], []
    patient = 0
    for _ in range(EPOCHS):

        total_loss = 0
        # Training loop
        for step, batch in enumerate(train_dataloader):

            # add batch to gpu
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            model.zero_grad()

            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask, labels=b_labels)
            # get the loss
            loss = outputs[0]
            # Perform a backward pass to calculate the gradients.
            loss.backward()
            # track train loss
            total_loss += loss.item()
            # Clip the norm of the gradient
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=MAX_GRAD_NORM)
            # update parameters
            optimizer.step()
            # Update the learning rate.
            scheduler.step()

        avg_train_loss = total_loss / len(train_dataloader)

        model.eval()

        # Reset the validation loss for this epoch.
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        predictions , true_labels = [], []
        
        for batch in valid_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            # saving memory and speeding up validation
            with torch.no_grad():

                outputs = model(b_input_ids, token_type_ids=None,
                                attention_mask=b_input_mask, labels=b_labels)

            logits = outputs[1].detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            eval_loss += outputs[0].item()
            predictions.extend(np.argmax(logits, axis=1))
            true_labels.extend(label_ids)
            
        eval_loss = eval_loss / len(valid_dataloader)
        validation_loss_values.append(eval_loss)
        
        print(f'''Epoch [{_+1}/{EPOCHS}] total loss complete. Train Loss: {avg_train_loss:.5f}. Val Loss: {eval_loss:.5}''')

        pred_tags = [label_dict_inverse[int(p)] for p in predictions]
        valid_tags = [label_dict_inverse[int(l)] for l in true_labels]
        print("Validation Accuracy: {}".format(accuracy_score(pred_tags, valid_tags)))

        valid_acc.append(accuracy_score(pred_tags, valid_tags))   
        
        # condition setting (model saved)
        if accuracy_score(pred_tags, valid_tags)>=max(valid_acc) and eval_loss<=min(validation_loss_values): #,default=1e9): 
            patient = 0
            print("saving state dict")
            torch.save(model.state_dict(), f"risk_model.pt")
        else:
            # early stopping
            patient += 1
            if patient == PATIENT:
                print(f'Early Stop. Best Acc {max(valid_acc)}')
                break
        print() 

def evaluate(args):

    # load testing data
    df = pd.read_csv(args['data_dir'],sep=',')

    # load model saved before
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-chinese", 
        num_labels = LABEL_NUM, 
        output_attentions = False, 
        output_hidden_states = False, 
    ).cuda()
    model.load_state_dict(torch.load(args['model_file'])) 

    possible_labels = ['1','0']
    label_dict = {}
    for index, possible_label in enumerate(possible_labels):
        label_dict[possible_label] = index
    label_dict_inverse = {v: k for k, v in label_dict.items()}

    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    test_input_ids, test_attention_masks = tokenize(df['text'], MAX_LENGTH, tokenizer)

    batch_size = 1  

    # Create the DataLoader.
    prediction_data = TensorDataset(test_input_ids, test_attention_masks)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size, shuffle=False)

    # prediction step
    model.eval()

    predictions , probability = [], []

    for batch in prediction_dataloader:
        batch = tuple(t.to(device) for t in batch)

        b_input_ids, b_input_mask = batch

        with torch.no_grad():

            outputs = model(b_input_ids, token_type_ids=None, 
                        attention_mask=b_input_mask)

        logits = outputs[0]
        logits = logits.detach().cpu()
        preds_flat = label_dict_inverse[int(np.argmax(logits, axis=1).flatten())]

        prob = F.softmax(logits,dim=-1)
        probability.append(float(prob[0][0]))
        predictions.append(preds_flat)

    # output result 
    result = pd.DataFrame({'article_id':df['article_id'].values,'probability':probability})

    return result.to_csv('decision.csv', index=None)

if __name__ == '__main__':

    main()

    # training
    # python3 main.py --mode train --data_dir ../data/Train_risk_preprocess.csv

    # predict
    # python3 main.py --mode train --data_dir ../data/Train_risk_preprocess.csv --model_file qa_model.pt
