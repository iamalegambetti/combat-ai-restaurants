# Combat AI with AI: ounteract Machine-Generated Fake Restaurant Reviews on Social Media

This repository includes the replication code for modeling part of the paper "Combat AI with AI: Counteract Machine-Generated Fake Restaurant Reviews on Social Media". Paper available at [https://arxiv.org/abs/2302.07731](https://arxiv.org/abs/2302.07731).

In our implementation, Python 3.8 was used. 

The experiments.ipynb and experimentsV2.ipynb notebooks report the evaluation of the models. 

## Updates
11 April 2023: Added GPT-Neo version. 

## Data
The /data folder contains the partitions used to train, validate, and test the models.
For machine learning models, i.e. Logistic Regression, Naive Bayes, Random Forest, and XGBoost the splits /data/train.csv and /data/test.csv should be used, as 5-fold cross-validation is automatically performed by the model. 

For the deep learning models, i.e. BiLSTM and GPT-2, /data/train_val.csv, /data/val_val.csv, and /data/test.csv should be used for training, validation and testing, respectively. 

## Models

### Machine Learning Benchmarks 
Benchmark models are Logistic Regression, Naive Bayes, Random Forest, and XGBoost. For example, to run Logistic Regression use:
```
python3 src/benchmarks_sklearn.py --model lr
```
Summary table of models available and command to use:
| Model               | Parameter|
|---------------------|----------|
| Logistic Regression | lr       |
| Naive Bayes         | nb      |
| Random Forest       | rf      |
| XGBoost             | xgb     |

Optionally, a "--save" argument can be passed to save the model weights (default "False").
Here, train and test is automatically performed. 

### Deep Learning Models
#### GPT
Deep learning models are BiLSTM and GPT-2, and GPTNeo (open source alias for GPT-3).

To train GPT Models use:
```
python3 src/gpt/train.py --model_version "specific model version"
```
Summary table of models available and command to use:
| Model Version       | --model_version|
|---------------------|----------|
| GPTNeo 125M (default) |        EleutherAI/gpt-neo-125M |
| GPT2 117M         | gpt2      |



The script can be parametrized by setting the learning rate "-lr" (default 1e-04), and whether optimizing for the Youden's J statistics "--youden" (default True).
The model automatically saves the weights at each best epoch in the /output folder. 


##### Weights
| Model Version       | Weights|
|---------------------|----------|
| GPTNeo 125M (default) |  [Here](https://drive.google.com/file/d/1OrweZO9L9nTmkGjHMIT_LHu9FVc_y6bM/view?usp=sharing)    |
| GPT2 117M         |   [Here](https://drive.google.com/file/d/1-C86zgQ8DjDWU8VJ75jP7DHwGMgrSi-d/view?usp=sharing)     |

Store them in the /output folder. 

Evaluation:
```
python3 src/gpt/evaluate.py --model_version "specific model version" --weights_path "specific weights path"
```
The scripts automatically uses the best GPTNeo weights provided as above. However, you can change them at your needs by modifiying the "--weights_path /path_to_the_weights_you_want_to_use". You can also set a different classification threshold "-j 0.x" (default 0.5708).



#### BiLSTM
IMPORTANT:
BiLSTM requires the the installation of torchtext==0.10.0. Thus, a downgraded general version of torch may be needed. We used torch==1.9.0. Train and test are performed simoultaneously. 

```
python3 src/lstm.py
```
The script can be parametrized by setting the learning rate "-lr" (default 1e-04).

## Cite Us
```bibtex
@misc{gambetti2023fakereviews,
  author = {Gambetti, Alessandro and Han, Qiwei},
  title = {Combat AI With AI: Counteract Machine-Generated Fake Restaurant Reviews on Social Media},
  publisher = {arXiv},
  year = {2023},
  keywords = {Computation and Language (cs.CL), Artificial Intelligence (cs.AI), Information Retrieval (cs.IR), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  doi = {10.48550/ARXIV.2302.07731},
  url = {https://arxiv.org/abs/2302.07731},
  copyright = {Creative Commons Attribution 4.0 International}
}
```