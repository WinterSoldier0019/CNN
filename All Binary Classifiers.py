# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# Importing the Header Files

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# %%
Data = pd.read_csv("Churn_Modelling.csv")


# %%
Data.head(n=5)


# %%
Data.shape


# %%
Data.describe(include = "all")


# %%
Data = Data.drop(labels=["RowNumber", "CustomerId", "Surname"],axis = 1)
Data.head()


# %%
from sklearn.preprocessing import LabelEncoder as LL


# %%
le1 = LL()
le2 = LL()


# %%
Data["Gender"] = le1.fit_transform(Data["Gender"])


# %%
Data.head()


# %%
Data["Geography"] = le2.fit_transform(Data["Geography"])
Data.head()


# %%
from sklearn.preprocessing import OneHotEncoder as OHE


# %%
x = Data.iloc[:, 0:10]
y = Data.iloc[:, -1]


# %%
x.head()


# %%
y.head()


# %%
ohe = OHE(sparse=False, categorical_features= [1])
x = ohe.fit_transform(x)


# %%
x


# %%
x_dummy = pd.DataFrame(x)
x_dummy.head()


# %%
x_dummy = x_dummy.drop(labels=[1],axis=1)
x_dummy.head()


# %%
from sklearn.preprocessing import StandardScaler as SS
Scaler = SS()
x_scaled = Scaler.fit_transform(x_dummy)
x_scaled


# %%
y


# %%
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, random_state=42, test_size = 0.2)


# %%
from sklearn.linear_model import LogisticRegression as LR
regressor = LR()
regressor.fit(x_train,y_train)


# %%
regressor.score(x_train,y_train)


# %%
y_pred = regressor.predict(x_test)


# %%
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
cm


# %%
print((1543 + 79)/2000)


# %%
from sklearn.neighbors import KNeighborsClassifier
regressorknn = KNeighborsClassifier()
regressorknn.fit(x_test,y_test)


# %%
regressorknn.score(x_train,y_train)


# %%
y_knn = regressorknn.predict(x_test)


# %%
cm_knn = confusion_matrix(y_test,y_knn)
cm_knn


# %%
print((1565 + 170)/2000)


# %%
from sklearn.svm import SVC
regressorsvc = SVC()
regressorsvc.fit(x_train,y_train)


# %%
regressorsvc.score(x_train,y_train)


# %%
y_svc = regressorsvc.predict(x_test)
cm_svc = confusion_matrix(y_test,y_svc)
cm_svc


# %%
print((1569 + 147)/2000)


# %%
from sklearn.naive_bayes import GaussianNB
regressornvb = GaussianNB()
regressornvb.fit(x_train,y_train)


# %%
regressornvb.score(x_train,y_train)


# %%
y_nvb = regressornvb.predict(x_test)


# %%
cm_nvb = confusion_matrix(y_test,y_nvb)
cm_nvb


# %%
print((1561 + 105)/2000)


# %%
from sklearn.tree import DecisionTreeClassifier
regressorTree = DecisionTreeClassifier()
regressorTree.fit(x_train,y_train)


# %%
regressorTree.score(x_train,y_train)


# %%
y_dec = regressorTree.predict(x_test)
cm_tree = confusion_matrix(y_test,y_dec)
cm_tree


# %%
print((1358 + 204)/2000)


# %%
from sklearn.ensemble import RandomForestClassifier
regressorforest = RandomForestClassifier()
regressorforest.fit(x_train,y_train)


# %%
regressorforest.score(x_train,y_train)


# %%
y_forest = regressorforest.predict(x_test)
cm_forest = confusion_matrix(y_test,y_forest)
cm_forest


# %%
print((1540 + 159)/2000)

# %% [markdown]
# Making ANN

# %%
import keras


# %%
from keras.models import Sequential
from keras.layers import Dense


# %%
classifier = Sequential()


# %%
classifier.add(Dense(units=40, activation="relu", input_dim = 11))
classifier.add(Dense(units=40, activation="relu"))
classifier.add(Dense(units=40, activation="relu"))
classifier.add(Dense(units=1, activation="sigmoid"))


# %%
classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics= ["accuracy"])


# %%
classifier.fit(x_train, y_train, batch_size = 32, epochs = 100)


# %%
y_ann = classifier.predict(x_test)
y_ann = (y_ann > 0.5)
cm_ann = confusion_matrix(y_test,y_ann)
cm_ann


# %%
print((1486 + 178)/2000)


# %%


