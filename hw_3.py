import pandas as pd
import numpy as np
eps = np.finfo(float).eps
from numpy import log2 as log
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus

# train_data = {'Home': [1,0,1,1,0,1,1,0,0,1,0,0,0,1,1,1,1,0,1,1,1,1,1,0],
        # 'Top25': [0,0,1,0,1,0,1,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,1,1],
        # 'NBC': [1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0],
        # 'ESPN': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        # 'FOX': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # 'ABC': [0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
        # 'CBS': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        # 'Win': [1,1,1,1,0,1,1,1,1,1,1,0,0,1,0,0,1,0,0,1,0,1,0,0]
        # }

# test_data = {'Home': [1,1,0,0,1,0,1,1,1,0,1,0],
        # 'Top25': [0,1,0,0,0,0,1,0,0,1,0,1],
        # 'NBC': [1,1,0,0,1,0,1,1,1,0,1,0],
        # 'ESPN': [0,0,1,0,0,0,0,0,0,0,0,0],
        # 'FOX': [0,0,0,1,0,0,0,0,0,0,0,0],
        # 'ABC': [0,0,0,0,0,1,0,0,0,1,0,1],
        # 'CBS': [0,0,0,0,0,0,0,0,0,0,0,0],
        # 'Win': [1,0,1,1,1,1,1,1,1,0,1,0]
        # }

test_train_data = {'Home': [1,0,1,1,0,1,1,0,0,1,0,0,0,1,1,1,1,0,1,1,1,1,1,0,1,1,0,0,1,0,1,1,1,0,1,0],
        'Top25': [0,0,1,0,1,0,1,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,1,1,0,1,0,0,0,0,1,0,0,1,0,1],
        'NBC': [1,0,1,1,0,1,1,0,0,1,1,0,0,1,1,1,0,0,1,1,0,1,1,0,1,1,0,0,1,0,1,1,1,0,1,0],
        'ESPN':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
        'FOX':[0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
        'ABC':[0,1,0,0,1,0,0,1,1,0,0,0,1,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,1],
        'CBS':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        'Win':[1,1,1,1,0,1,1,1,1,1,1,0,0,1,0,0,1,0,0,1,0,1,0,0,1,0,1,1,1,1,1,1,1,0,1,0]
        }

df_test_train = pd.DataFrame(data=test_train_data)

def main():
    feature_cols = ['Home', 'Top25', 'NBC', 'ESPN', 'FOX', 'ABC', 'CBS']
    X = df_test_train[feature_cols]
    y = df_test_train.Win
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=(14/38),random_state=None)
    clf = DecisionTreeClassifier(criterion="entropy")
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print('X_train',X_train)
    print('X_test',X_test)
    print(y_pred)
    print('Accuracy:',metrics.accuracy_score(y_test,y_pred))
    dot_data = StringIO()
    export_graphviz(clf, out_file=dot_data,
            filled=True, rounded=True,
            special_characters=True,feature_names = feature_cols,class_names=['0','1'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png('id3.png')
    Image(graph.create_png())

if __name__=="__main__":
    main()
