import numpy as np
from sklearn import ensemble
import sklearn.model_selection
import sklearn.metrics
import pandas as pd


df = pd.read_csv("data/crop_yield.csv", comment='#')
df.replace("?", -99999, inplace=True)
df.fillna(0, inplace=True)

y = np.asarray(df["hg/ha_yield"])

df.drop(["Num", "Area", "Year", "hg/ha_yield"], 1, inplace=True)
factorized, label = pd.factorize(df["Item"])
label = label.tolist()
df["Item"] = factorized

x = np.asarray(df)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

clf = ensemble.RandomForestRegressor()
clf.fit(x_train, y_train)

accuracy = sklearn.metrics.max_error(y_test, clf.predict(x_test))
print(accuracy)

data = np.asarray([label.index("Maize"), 1485, 121, 16.37]).reshape(1, -1)
print(clf.predict(data))
