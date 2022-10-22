import pandas as pd
import numpy as np

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

train_data = {
        'Home':     [1,0,1,1,0,1,1,0,0,1,0,0,0,1,1,1,1,0,1,1,1,1,1,0],
        'Top25':    [0,0,1,0,1,0,1,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,1,1],
        'NBC':      [1,0,1,1,0,1,1,0,0,1,1,0,0,1,1,1,0,0,1,1,0,1,1,0],
        'ESPN':     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
        'FOX':      [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
        'ABC':      [0,1,0,0,1,0,0,1,1,0,0,0,1,0,0,0,0,1,0,0,0,0,0,1],
        'CBS':      [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
        'Win':      [1,1,1,1,0,1,1,1,1,1,1,0,0,1,0,0,1,0,0,1,0,1,0,0]
        }

test_data = {
        'Home':     [1,1,0,0,1,0,1,1,1,0,1,0],
        'Top25':    [0,1,0,0,0,0,1,0,0,1,0,1],
        'NBC':      [1,1,0,0,1,0,1,1,1,0,1,0],
        'ESPN':     [0,0,1,0,0,0,0,0,0,0,0,0],
        'FOX':      [0,0,0,1,0,0,0,0,0,0,0,0],
        'ABC':      [0,0,0,0,0,1,0,0,0,1,0,1],
        'CBS':      [0,0,0,0,0,0,0,0,0,0,0,0],
        'Win':      [1,0,1,1,1,1,1,1,1,0,1,0]
        }

test_train_data = {
        'Home':     [1,0,1,1,0,1,1,0,0,1,0,0,0,1,1,1,1,0,1,1,1,1,1,0,1,1,0,0,1,0,1,1,1,0,1,0],
        'Top25':    [0,0,1,0,1,0,1,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,1,1,0,1,0,0,0,0,1,0,0,1,0,1],
        'NBC':      [1,0,1,1,0,1,1,0,0,1,1,0,0,1,1,1,0,0,1,1,0,1,1,0,1,1,0,0,1,0,1,1,1,0,1,0],
        'ESPN':     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
        'FOX':      [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
        'ABC':      [0,1,0,0,1,0,0,1,1,0,0,0,1,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,1],
        'CBS':      [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        'Win':      [1,1,1,1,0,1,1,1,1,1,1,0,0,1,0,0,1,0,0,1,0,1,0,0,1,0,1,1,1,1,1,1,1,0,1,0]
        }

df_train = pd.DataFrame(data=train_data)
df_test = pd.DataFrame(data=test_data)
df_test_train = pd.DataFrame(data=test_train_data)

def main():
    feature_cols = ['Home', 'Top25', 'NBC', 'ESPN', 'FOX', 'ABC', 'CBS']
    x_train = df_train[feature_cols]
    y_train = df_train.Win
    x_test = df_test[feature_cols]
    y_test = df_test.Win
    gnb = GaussianNB()
    y_pred = gnb.fit(x_train, y_train).predict(x_test)

    print('Predicition',y_pred)
    print('Accuracy:',metrics.accuracy_score(y_test,y_pred))
    print('Precision, Recall, F1',metrics.precision_recall_fscore_support(y_test,y_pred))

if __name__=="__main__":
    main()
