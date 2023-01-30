from model import GPT2
from transformers import GPT2Tokenizer, GPT2Model
import torch
import argparse
import numpy as np

# script arguments 
parser = argparse.ArgumentParser()
parser.add_argument('--weights_path', default='./output/gpt2_6_best.pt', required=False, type=str, help="Trained weights to evaluate.")
parser.add_argument('-j', default=0.6144000291824341, required=False, type=float, help='Thresold for optimisation. Defaults to 0.6414')
parser.add_argument("--text", required=False, default='review.txt', type=str, help='.txt file containing the review to perform inference on.')
args = parser.parse_args()

print('Loading model..')
weights_path = args.weights_path
j = args.j 
text = args.text
backbone_type = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(backbone_type)
backbone = GPT2Model.from_pretrained(backbone_type)
model = GPT2(backbone, last_hidden_size=768)
model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu'))) 
model.eval()
print('Model loaded.')

print('Predicting for review:')
text = open(text, 'r').read()
print(text)
encoded_input = tokenizer(text, return_tensors='pt')['input_ids']
pred = torch.sigmoid(model(encoded_input.unsqueeze(0)))
lab = np.where(pred > j, 1, 0)
print()

if lab == 1:
    print(f'Prediction output: FAKE. Probability: {pred}')
else:
    print(f'Prediction output: REAL. Probability: {1-pred}')


