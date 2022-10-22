# hw_5_training.py
# used to train a decision tree classifier and knn classifier

import pandas as pd
import numpy as np
import pickle

from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

clf = DecisionTreeClassifier(random_state=0)
neighbor = KNeighborsClassifier(n_neighbors=3)

def dcttrain(train_features, train_labels, filename):
    ret = clf.fit(train_features, train_labels)
    pickle.dump(clf, open(filename, 'wb'))
    return ret

def knntrain(train_features, train_labels, neighbors, filename):
    neighbor.n_neighbors = neighbors
    ret = neighbor.fit(train_features, train_labels)
    pickle.dump(neighbor, open(filename, 'wb'))
    return ret
