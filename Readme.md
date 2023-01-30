# Combat AI with AI: ounteract Machine-Generated Fake Restaurant Reviews on Social Media [Classifiers Replication Code]

This repository includes the replication code for modeling part of the paper "Combat AI with AI: ounteract Machine-Generated Fake Restaurant Reviews on Social Media".

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
#### GPT-2
Deep learning models are BiLSTM and GPT-2. 

To train the GPT-2 use:
```
python3 src/gpt_2/train.py 
```
The script can be parametrized by setting the learning rate "-lr" (default 1e-04), and whether optimizing for the Youden's J statistics "--youden" (default True).
The model automatically saves the weights at each best epoch in the /output folder. 

Also, we provide GPT-2 best weights to directly perform evaluation and inference. Download [here](https://drive.google.com/file/d/1-UysTaJ_qW3baj8NL05IujJW-FI66pQI/view?usp=sharing). Store them in the /output folder. 

Evaluation:
```
python3 src/gpt_2/evaluate.py
```
The scripts automatically uses the best weights found provided as above. However, you can change them at your needs by modifiying the "--weights_path /path_to_the_weights_you_want_to_use". You can also set a different classification threshold "-j 0.x" (default 0.6408).

Inference:
Store a restaurant review in the file: review.txt. Run: 
```
python3 src/gpt_2/inference.py
```
Here, you can change the weights path and j as in the evaluation script. 


#### BiLSTM
IMPORTANT:
BiLSTM requires the the installation of torchtext==0.10.0. Thus, a downgraded general version of torch may be needed. We used torch==1.9.0. Train and test are performed simoultaneously. 

```
python3 src/lstm.py
```
The script can be parametrized by setting the learning rate "-lr" (default 1e-04).