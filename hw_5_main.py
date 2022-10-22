# hw_5_main.py
# used to split a operate the trian a test for the iris dataset

import pandas as pd
import numpy as np
import os.path
from os import path

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from hw_5_training import dcttrain
from hw_5_training import knntrain
from hw_5_testing import dctscore
from hw_5_testing import knnscore

df = pd.read_csv('data/iris.data')
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'Class']
knnscores = []
knnkeys = []

def findbestk(train_features, train_labels, test_features, test_labels):
    for i in range(100):
        knntrain(train_features, train_labels, i+1, 'topk.sav')
        knnscores.append(knnscore(test_features, test_labels, 'topk.sav'))
        knnkeys.append(i)
        print(knnscores[i])


def main():
    feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    target_cols = df.Class

    neighbor = KNeighborsClassifier(n_neighbors=3)
    clf = DecisionTreeClassifier(random_state=0)


    part1 = np.array_split(df.sample(frac=1),5)[0]
    part2 = np.array_split(df.sample(frac=1),5)[1]
    part3 = np.array_split(df.sample(frac=1),5)[2]
    part4 = np.array_split(df.sample(frac=1),5)[3]
    part5 = np.array_split(df.sample(frac=1),5)[4]

    train1 = pd.concat([part2,part3,part4,part5])
    train2 = pd.concat([part1,part3,part4,part5])
    train3 = pd.concat([part1,part2,part4,part5])
    train4 = pd.concat([part1,part2,part3,part5])
    train5 = pd.concat([part1,part2,part3,part4])

    df_train1 = pd.DataFrame(data=train1)
    df_train1x = df_train1[feature_cols]
    df_train1y = df_train1.Class

    df_train2 = pd.DataFrame(data=train2)
    df_train2x = df_train2[feature_cols]
    df_train2y = df_train2.Class


    df_train3 = pd.DataFrame(data=train3)
    df_train3x = df_train3[feature_cols]
    df_train3y = df_train3.Class


    df_train4 = pd.DataFrame(data=train4)
    df_train4x = df_train4[feature_cols]
    df_train4y = df_train4.Class

    df_train5 = pd.DataFrame(data=train5)
    df_train5x = df_train5[feature_cols]
    df_train5y = df_train5.Class


    df_test1 = pd.DataFrame(data=part1)
    df_test1x = df_test1[feature_cols]
    df_test1y = df_test1.Class

    df_test2 = pd.DataFrame(data=part2)
    df_test2x = df_test2[feature_cols]
    df_test2y = df_test2.Class

    df_test3 = pd.DataFrame(data=part3)
    df_test3x = df_test3[feature_cols]
    df_test3y = df_test3.Class

    df_test4 = pd.DataFrame(data=part4)
    df_test4x = df_test4[feature_cols]
    df_test4y = df_test4.Class

    df_test5 = pd.DataFrame(data=part5)
    df_test5x = df_test5[feature_cols]
    df_test5y = df_test5.Class


    knnfilename = 'knntrain.sav'
    dctfilename = 'dcttrain.sav'

    findbestk(df_train1x, df_train1y, df_test1x, df_test1y)
    findbestk(df_train2x, df_train2y, df_test2x, df_test2y)
    findbestk(df_train3x, df_train3y, df_test3x, df_test3y)
    findbestk(df_train4x, df_train4y, df_test4x, df_test4y)
    findbestk(df_train5x, df_train5y, df_test5x, df_test5y)

    # plt.hist(knnscores, bins=100)
    plt.bar(knnkeys, knnscores)
    plt.show()

    if not path.exists(knnfilename):
        knntrain(df_train1x, df_train1y, 11, knnfilename)
    if not path.exists(dctfilename):
        dcttrain(df_train1x, df_train1y, dctfilename)

    dtc_avg = dctscore(df_test1x, df_test1y, dctfilename)
    dtc_avg = dtc_avg + dctscore(df_test2x, df_test2y, dctfilename)
    dtc_avg = dtc_avg + dctscore(df_test3x, df_test3y, dctfilename)
    dtc_avg = dtc_avg + dctscore(df_test4x, df_test4y, dctfilename)
    dtc_avg = dtc_avg + dctscore(df_test5x, df_test5y, dctfilename)
    dtc_avg = dtc_avg/5
    print('dtc_avg', dtc_avg)

    knn_avg = knnscore(df_test1x, df_test1y, knnfilename)
    knn_avg = knn_avg + knnscore(df_test2x, df_test2y, knnfilename)
    knn_avg = knn_avg + knnscore(df_test3x, df_test3y, knnfilename)
    knn_avg = knn_avg + knnscore(df_test4x, df_test4y, knnfilename)
    knn_avg = knn_avg + knnscore(df_test5x, df_test5y, knnfilename)
    knn_avg = knn_avg/5
    print('knn_avg',knn_avg)

if __name__=="__main__":
    main()
