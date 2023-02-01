import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import argparse

# script arguments 
parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, type=str, help="Available models: 'lr', 'nb', 'rf', 'xgb' .")
parser.add_argument("--save", required=False, default=False, type=bool, help="Default to false.")
args = parser.parse_args()

def load_data(path_train, path_test):
    train = pd.read_csv(path_train)
    test = pd.read_csv(path_test)
    return train, test

def transform_data(train, test):
    cvec = CountVectorizer(stop_words = 'english', min_df = 0.0001) # filter out 0.001% of non recurrent words
    cvec.fit(train.text)
    X_train = cvec.transform(train.text)
    y_train = train.label
    X_test = cvec.transform(test.text)
    y_test = test.label
    return X_train, y_train, X_test, y_test


models = {"lr":{'model':LogisticRegression(max_iter=2000), 'params':{"C":[0.001, 0.01, 0.1, 1, 10, 100]}},
    'nb':{'model':MultinomialNB(), 'params':{"alpha":[0.001, 0.01, 0.1, 1, 10, 100]}},
    'rf':{'model':RandomForestClassifier(), 'params':{"n_estimators":[75, 100, 125], 
        "max_depth":[12, 14, 16], "min_samples_split":[2,3,4]}},
    'xgb':{'model':XGBClassifier(), 'params':{"eta":[0.01, 0.03, 0.05], "max_depth":[4,6,8], 
        "n_estimators": [75, 100, 125]}}
}

def optimise(model_name:str, X_train, y_train, save):
    obj = models.get(model_name)
    model = obj.get('model')
    params = obj.get('params')
    clf = GridSearchCV(model, params, cv=5)
    clf.fit(X_train, y_train)
    print('Best estimator: ')
    print(clf.best_estimator_)
    if save:
        with open(f'./output/{model_name}.pickle','wb') as f:
            pickle.dump(clf,f)
    return clf.best_estimator_

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

def predict(model, X_test):
    pred = model.predict(X_test)
    return pred

def pipeline_sklearn(model_name, path_train="./data/train.csv", path_test="./data/test.csv", save = args.save):
    if model_name not in list(models.keys()):
        raise ValueError(f'Model invalid. Picke one among: {list(models.keys())}.')
    print(f'Model chosen: {model_name}.')
    print('Optimising..')
    train, test = load_data(path_train, path_test)
    X_train, y_train, X_test, y_test = transform_data(train, test)
    best_estimator = optimise(model_name, X_train, y_train, save)
    print(best_estimator)
    pred = predict(best_estimator, X_test)
    metrics(pred, y_test)

if __name__ == '__main__':
    path_train = "./data/train.csv"
    path_test = "./data/test.csv"
    model_name = args.model
    pipeline_sklearn(model_name, path_train, path_test)