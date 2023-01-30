#! pip install torchtext==0.10.0

import torch
import pandas as pd
import numpy as np
from torchtext import data
import torchtext
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score
import argparse
print('LSTM')

parser = argparse.ArgumentParser()
parser.add_argument("-lr", default=0.0001, required=False, type=float, help="Learning rate.")
args = parser.parse_args()

# load data
TEXT = torchtext.legacy.data.Field(tokenize='spacy',batch_first=True,include_lengths=True)
LABEL = torchtext.legacy.data.LabelField(dtype = torch.float,batch_first=True)
fields = [('text',TEXT), ("label",LABEL)]

# load training data 
training_data = torchtext.legacy.data.TabularDataset(path="./data/train_val.csv",
                                    format="csv",
                                    fields=fields,
                                    skip_header=True
                                   )
test_data = torchtext.legacy.data.TabularDataset(path="./data/val_val.csv",
                                    format="csv",
                                    fields=fields,
                                    skip_header=True
                                   )

# Building vocabularies 
TEXT.build_vocab(training_data,
                 min_freq=5)
LABEL.build_vocab(training_data)

# some params 
device = torch.device("cuda")
BATCH_SIZE = 64

# Iterators
train_iterator,validation_iterator = torchtext.legacy.data.BucketIterator.splits(
    (training_data,test_data),
    batch_size = BATCH_SIZE,
    sort_key = lambda x:len(x.text),
    sort_within_batch = True,
    device = device
)

# Model 
class LSTMNet(nn.Module):
    
    def __init__(self,vocab_size,embedding_dim,hidden_dim,output_dim,n_layers,bidirectional,dropout):
        super(LSTMNet,self).__init__()
        self.embedding = nn.Embedding(vocab_size,embedding_dim)
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers = n_layers,
                            bidirectional = bidirectional,
                            dropout = dropout,
                            batch_first = True
                           )
        self.fc = nn.Linear(hidden_dim * 2,output_dim)
        self.sigmoid = nn.Sigmoid()
        
    
    def forward(self,text,text_lengths):
        embedded = self.embedding(text)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.cpu(),batch_first=True)
        packed_output,(hidden_state,cell_state) = self.lstm(packed_embedded)
        hidden = torch.cat((hidden_state[-2,:,:], hidden_state[-1,:,:]), dim = 1)
        dense_outputs=self.fc(hidden)
        outputs=self.sigmoid(dense_outputs)
        return outputs

# hyperparams
SIZE_OF_VOCAB = len(TEXT.vocab)
EMBEDDING_DIM = 100
NUM_HIDDEN_NODES = 64
NUM_OUTPUT_NODES = 1
NUM_LAYERS = 2
BIDIRECTION = True
DROPOUT = 0.2

model = LSTMNet(SIZE_OF_VOCAB,
                EMBEDDING_DIM,
                NUM_HIDDEN_NODES,
                NUM_OUTPUT_NODES,
                NUM_LAYERS,
                BIDIRECTION,
                DROPOUT
               )

model = model.to(device)
optimizer = optim.AdamW(model.parameters(),lr=args.lr)
criterion = nn.BCELoss()
criterion = criterion.to(device)

def binary_accuracy(preds, y):
    rounded_preds = torch.round(preds)    
    correct = (rounded_preds == y).float() 
    acc = correct.sum() / len(correct)
    return acc

def metrics(preds, y):
    rounded_preds = torch.round(preds)
    y = y.detach().cpu().numpy()
    rounded_preds = rounded_preds.detach().cpu().numpy()
    acc = accuracy_score(y, rounded_preds)
    prec = precision_score(y, rounded_preds)
    rec = recall_score(y, rounded_preds)
    return {'acc':acc, 'prec':prec, 'rec':rec}

def train(model,iterator,optimizer,criterion):
    
    epoch_loss = 0.0
    epoch_acc = 0.0
    
    model.train()
    
    for batch in iterator:
        
        optimizer.zero_grad()
        text,text_lengths = batch.text
        predictions = model(text,text_lengths).squeeze()
        loss = criterion(predictions,batch.label)
        loss.backward()
        acc = binary_accuracy(predictions,batch.label)
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)



def evaluate(model,iterator,criterion):
    
    epoch_loss = 0.0
    epoch_acc = 0.0
    accs = []
    precs = []
    recs = []
    model.eval()
    
    with torch.no_grad():
        for batch in iterator:
            text,text_lengths = batch.text
            
            predictions = model(text,text_lengths).squeeze()
              
            #compute loss and accuracy
            loss = criterion(predictions, batch.label)
            acc = binary_accuracy(predictions, batch.label)

            # my line
            met = metrics(predictions, batch.label)
            precs.append(met['prec'])
            recs.append(met['rec'])
            accs.append(met['acc'])
            
            #keep track of loss and accuracy
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator), np.mean(precs), np.mean(recs)

EPOCH_NUMBER = 100
best_acc = 0
count = 0
for epoch in range(1,EPOCH_NUMBER+1):
    
    train_loss,train_acc = train(model,train_iterator,optimizer,criterion)
    valid_loss,valid_acc, prec, rec, accs = evaluate(model,validation_iterator,criterion)
    # Showing statistics
    print('Epoch: ', epoch)
    print(accs, prec, rec)
    print()
    count += 1
    if accs > best_acc:
        count = 0
        best_acc = accs
    
    if count == 5:
        print('Stopping at epoch: ', epoch, '.')
        break

test_test_data = torchtext.legacy.data.TabularDataset(path="./data/test.csv",
                                    format="csv",
                                    fields=fields,
                                    skip_header=True
                                   )

train_iterator,test_iterator = torchtext.legacy.data.BucketIterator.splits(
    (training_data,test_test_data),
    batch_size = BATCH_SIZE,
    sort_key = lambda x:len(x.text),
    sort_within_batch = True,
    device = device
)

out_test = evaluate(model,test_iterator,criterion)
print('Converged. Results: ')
print(out_test)

