import pandas as pd
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score



df = pd.read_csv('data/iris.data')
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'Class']


def main():
    feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    target_cols = df.Class

    neighbor = KNeighborsClassifier(n_neighbors=3)

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

    df_train1 = pd.DataFrame(data=train_1)
    df_train1x = df_train1[feature_cols]
    df_train1y = df_train1.Class

    df_test1 = pd.DataFrame(data=part1)
    df_test1x = df_test1[feature_cols]

    neighbor.fit(df_train1x,df_train1y)

    predict = neighbor.predict(df_test1x)

    print(part1.Class)
    print(neighbor.score(df_train_1x, df_train_1y))
    print(accuracy_score(predict, part1.Class))

if __name__=="__main__":
    main()
