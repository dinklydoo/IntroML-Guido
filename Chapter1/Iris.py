from pandas.core.common import not_none
from pkg_resources import non_empty_lines

from sklearn.datasets import load_iris
import mglearn
import matplotlib.pyplot as plt
import numpy as np
iris_dataset = load_iris()

import pandas as pd
from sklearn.model_selection import train_test_split
X_train, X_text, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state = 0)

iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
pd.plotting.scatter_matrix(iris_dataframe, c = y_train, figsize=(15,15),
                           marker = 'o', hist_kwds={'bins':20}, s = 60,
                           alpha =.8, cmap = mglearn.cm3)
# plt.show()
# display scatter plots

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(X_train, y_train)
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric = 'minkowski',
                     metric_params=None, n_jobs=None, n_neighbors=1, p=2, weights='uniform')

X_new = np.array([[5,2.9,1,0.2]])
# Note : scikit always expects 2d - array for input


prediction = knn.predict(X_new)
print("Prediction", prediction)
print("Predicted target name", iris_dataset.target_names[prediction])