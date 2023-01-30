from model import GPT2
from dataset import FakeReviewsDataset
from torch.utils.data import DataLoader
from transformers import GPT2Model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np
import argparse

# script arguments 
parser = argparse.ArgumentParser()
parser.add_argument('--weights_path', default='./output/gpt2_6_best.pt', required=False, type=str, help="Trained weights to evaluate.")
parser.add_argument('-j', default=0.6144000291824341, required=False, type=float, help='Thresold for optimisation. Defaults to 0.6414')
args = parser.parse_args()

print('Loading Model..')
backbone_type = "gpt2"
device = 'cpu'
j=args.j
weights_path = args.weights_path
dataset = FakeReviewsDataset("./data/test.csv", backbone_type)
test_dataloader = DataLoader(dataset, 1)
backbone = GPT2Model.from_pretrained(backbone_type)
model = GPT2(backbone, last_hidden_size=768)
model.load_state_dict(torch.load(weights_path, map_location=torch.device(device))) 
model.eval()
print('Model Loaded.')

def metrics(pred, true):
    acc = round(accuracy_score(true, pred) * 100, 2)
    prec = round(precision_score(true, pred) * 100, 2)
    rec = round(recall_score(true, pred) * 100, 2)
    f1 = round(f1_score(true, pred) * 100, 2)
    print('Evaluation metrics on test set: ')
    print('Accuracy: ', acc, "%")
    print('Precision: ', prec, "%")
    print('Recall: ', rec, "%")
    print('F1-score', f1, "%")

def evaluate_gpt(j = j):
    print('Evaluating..')
    with torch.no_grad():
        predictions = []
        true = []
        for data_test in test_dataloader:
            X_test = data_test['input']['input_ids'].to(device)
            y_test = data_test['label'].numpy()
            y_pred_test = F.sigmoid(model(X_test).unsqueeze(0)).squeeze(0)#.detach().cpu().numpy()
            y_pred_test = np.where(y_pred_test > j, 1, 0)
            predictions.append(y_pred_test)
            true.append(y_test)    
    metrics(predictions, true)


def main():
    evaluate_gpt()

if __name__ == '__main__':
    main()
