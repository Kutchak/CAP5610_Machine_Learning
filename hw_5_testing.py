# hw_5_testing.py
# used to test the classifiers trained from training.py

import pandas as pd
import numpy as np
import pickle

from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def dctscore(test_features, test_labels, filename):
    loaded_model = pickle.load(open(filename, 'rb'))
    result = loaded_model.score(test_features, test_labels)
    return result

def knnscore(test_features, test_labels, filename):
    loaded_model = pickle.load(open(filename, 'rb'))
    result = loaded_model.score(test_features, test_labels)
    return result

    # print(neighbor.score(df_train1x, df_train1y))
    # print('test 1 knn accuracy score', accuracy_score(predict1_knn, part1.Class))
    # print('test 1 dct accuracy score', accuracy_score(predict1_dct, part1.Class))

    # print(neighbor.score(df_train2x, df_train2y))
    # print('test 2 knn accuracy score', accuracy_score(predict2_knn, part2.Class))
    # print('test 2 dct accuracy score', accuracy_score(predict2_dct, part2.Class))

    # print(neighbor.score(df_train3x, df_train3y))
    # print('test 3 knn accuracy score', accuracy_score(predict3_knn, part3.Class))
    # print('test 3 dct accuracy score', accuracy_score(predict3_dct, part3.Class))

    # print(neighbor.score(df_train4x, df_train4y))
    # print('test 4 knn accuracy score', accuracy_score(predict4_knn, part4.Class))
    # print('test 4 dct accuracy score', accuracy_score(predict4_dct, part4.Class))

    # print(neighbor.score(df_train5x, df_train5y))
    # print('test 5 knn accuracy score', accuracy_score(predict5_knn, part5.Class))
    # print('test 5 dct accuracy score', accuracy_score(predict5_dct, part5.Class))
