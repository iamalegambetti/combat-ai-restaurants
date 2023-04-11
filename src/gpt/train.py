import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from dataset import FakeReviewsDataset
from model import GPT2, GPTNeo
from torch.utils.data import DataLoader
import os
import numpy as np
from transformers import GPT2Model, GPTNeoForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve
import argparse

# script arguments 
parser = argparse.ArgumentParser()
parser.add_argument("-lr", default=0.0001, required=False, type=float, help="Learning rate.")
parser.add_argument("--youden", default=True, required=False, type=bool, help="Set to True for Youden J Statistics optimization.")
parser.add_argument("--model_version", default='EleutherAI/gpt-neo-125M', required=False, type=str, help="Backbone version of GPT models available.")
args = parser.parse_args()

def load_data(train, path = "./data/"):
    if train:
        train_data = FakeReviewsDataset(os.path.join(path, 'train_val.csv'), backbone_type)
        train_dataloader = DataLoader(train_data, 1)
        return train_dataloader
    else:
        test_data = FakeReviewsDataset(os.path.join(path, 'val_val.csv'), backbone_type)
        test_dataloader = DataLoader(test_data, 1)
        return test_dataloader

# model 
backbone_type = args.model_version

if backbone_type == 'EleutherAI/gpt-neo-125M':
    last_hidden_size = 768
    backbone = GPTNeoForSequenceClassification.from_pretrained(backbone_type)
    model = GPTNeo(backbone, last_hidden_size)
    backbone_type_save = 'gpt-neo-125M'

if backbone_type == 'gpt2-large':
    last_hidden_size = 1280
    backbone = GPT2Model.from_pretrained(backbone_type)
    model = GPT2(backbone, last_hidden_size) 
    backbone_type_save = backbone_type

if backbone_type == 'gpt2':
    last_hidden_size = 768
    backbone = GPT2Model.from_pretrained(backbone_type)
    model = GPT2(backbone, last_hidden_size) 
    backbone_type_save = backbone_type


# loss
criterion = nn.BCEWithLogitsLoss()

def sensivity_specifity_cutoff(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    idx = np.argmax(tpr - fpr)
    return thresholds[idx]

def train_gpt(EPOCHS=100, lr = args.lr):

    optimiser = torch.optim.AdamW(model.parameters(), lr=lr)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Backbone type:', backbone_type)
    print('Training on device: ', device)
    print('Learning rate:', args.lr)

    model.to(device)
    train_dataloader = load_data(train=True)
    test_dataloader = load_data(train=False)
    best = 0.0

    scheduler = StepLR(optimiser, step_size=5, gamma=0.1)
    count = 0
    for epoch in range(1, EPOCHS+1):
        print(f'EPOCH: {epoch}..')
        print('Training..')
        model.train()
        for i, data in enumerate(train_dataloader, start = 1):
            if i % 500 == 0: print(f'Processed {i}th batch..')
            optimiser.zero_grad()
            X = data['input']['input_ids'].to(device)
            y = data['label'].to(device).float()
            y_pred = model(X).unsqueeze(0)
            loss = criterion(y_pred, y)
            loss.backward()
            optimiser.step()
        
        model.eval()        
        with torch.no_grad():
            predictions = []
            probas = []
            true = []
            print('Testing..')
            for data_test in test_dataloader:
                X_test = data_test['input']['input_ids'].to(device)
                y_test = data_test['label'].numpy()
                y_pred_proba = F.sigmoid(model(X_test).unsqueeze(0)).squeeze(0).detach().cpu().numpy()
                y_pred_test = np.where(y_pred_proba > 0.5, 1, 0)
                predictions.append(y_pred_test)
                probas.append(y_pred_proba)
                true.append(y_test)
            
            true = np.array(true)
            true = true.reshape(-1)
            probas = np.array(probas)
            probas = probas.reshape(-1)
            j = sensivity_specifity_cutoff(true, probas)

            print('Prediction metrics at 0.5: ')
            accuracy = accuracy_score(true, predictions)
            precision = precision_score(true, predictions)
            recall = recall_score(true, predictions)
            print(f'Accuracy: {accuracy}')
            print(f'Precision: {precision}')
            print(f'Recall: {recall}')
            print()

            print(f'Prediction metrics at Youden {round(j, 4)}: ')
            probas = np.array(probas)
            
            predictions_you = np.where(probas >= j, 1, 0)
            accuracy_you = accuracy_score(true, predictions_you)
            precision_you = precision_score(true, predictions_you)
            recall_you = recall_score(true, predictions_you)
            print(f'Accuracy J: {accuracy_you}')
            print(f'Precision J: {precision_you}')
            print(f'Recall J: {recall_you}')

            count += 1
            accs_to_optimize = accuracy_you if args.youden else accuracy
            if accs_to_optimize > best:
                count = 0
                print('Found best accuracy. Saving to disk.')
                torch.save(model.state_dict(), f'./output/{backbone_type_save}_{epoch}.pt')
                best = accuracy_you 
            
        if count == 10:
            print('Early stopping at epoch: ', epoch, '.')
            break
        
        scheduler.step()
        print()
    
if __name__ == '__main__':
    train_gpt(1000)