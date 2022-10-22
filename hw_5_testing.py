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
