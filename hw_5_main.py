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

def findbestk(train_features, train_labels, test_features, test_labels):
    for i in range(100):
        knntrain(train_features, train_labels, i+1, 'topk.sav')
        knnscores.append(knnscore(test_features, test_labels, 'topk.sav'))
        print(knnscores[i])

    plt.bar(range(len(knnscores)), knnscores)
    plt.show()

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

    # if not path.exists(knnfilename):
        # knntrain(df_train1x, df_train1y, 4, knnfilename)
    # if not path.exists(dctfilename):
        # dcttrain(df_train1x, df_train1y, dctfilename)

    # dtcscore = dctscore(df_test1x, df_test1y, dctfilename)
    # print(dtcscore)

    # neighbor.fit(df_train1x, df_train1y)
    # clf.fit(df_train1x, df_train1y)

    # predict1_knn = neighbor.predict(df_test1x)
    # predict1_dct = clf.predict(df_test1x)

    # # print(part1.Class)
    # # print(neighbor.score(df_train1x, df_train1y))
    # print('test 1 knn accuracy score', accuracy_score(predict1_knn, part1.Class))
    # print('test 1 dct accuracy score', accuracy_score(predict1_dct, part1.Class))

    # df_test2 = pd.DataFrame(data=part2)
    # df_test2x = df_test2[feature_cols]

    # neighbor.fit(df_train2x,df_train2y)
    # clf.fit(df_train2x, df_train2y)

    # predict2_knn = neighbor.predict(df_test2x)
    # predict2_dct = clf.predict(df_test2x)

    # # print(part2.Class)
    # # print(neighbor.score(df_train2x, df_train2y))
    # print('test 2 knn accuracy score', accuracy_score(predict2_knn, part2.Class))
    # print('test 2 dct accuracy score', accuracy_score(predict2_dct, part2.Class))

    # df_test3 = pd.DataFrame(data=part3)
    # df_test3x = df_test3[feature_cols]

    # neighbor.fit(df_train3x,df_train3y)
    # clf.fit(df_train3x, df_train3y)

    # predict3_knn = neighbor.predict(df_test3x)
    # predict3_dct = clf.predict(df_test3x)

    # # print(part3.Class)
    # # print(neighbor.score(df_train3x, df_train3y))
    # print('test 3 knn accuracy score', accuracy_score(predict3_knn, part3.Class))
    # print('test 3 dct accuracy score', accuracy_score(predict3_dct, part3.Class))

    # df_test4 = pd.DataFrame(data=part4)
    # df_test4x = df_test4[feature_cols]

    # neighbor.fit(df_train4x,df_train4y)
    # clf.fit(df_train4x, df_train4y)

    # predict4_knn = neighbor.predict(df_test4x)
    # predict4_dct = clf.predict(df_test4x)

    # # print(part4.Class)
    # # print(neighbor.score(df_train4x, df_train4y))
    # print('test 4 knn accuracy score', accuracy_score(predict4_knn, part4.Class))
    # print('test 4 dct accuracy score', accuracy_score(predict4_dct, part4.Class))

    # df_test5 = pd.DataFrame(data=part5)
    # df_test5x = df_test5[feature_cols]

    # neighbor.fit(df_train5x,df_train5y)
    # clf.fit(df_train5x, df_train5y)

    # predict5_knn = neighbor.predict(df_test5x)
    # predict5_dct = clf.predict(df_test5x)

    # # print(part5.Class)
    # # print(neighbor.score(df_train5x, df_train5y))
    # print('test 5 knn accuracy score', accuracy_score(predict5_knn, part5.Class))
    # print('test 5 dct accuracy score', accuracy_score(predict5_dct, part5.Class))

if __name__=="__main__":
    main()
