
#Basic Libraries
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd


# Data Pre-processing Libraries
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split

# Modelling Libraries
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier

# Evaluation & CV Libraries
from sklearn.metrics import precision_score,accuracy_score

df=pd.read_csv('iris.csv')

df.head()

df.size,df.shape

df.info()

from sklearn.preprocessing import LabelEncoder
#Iris-setosa:0, Iris-versicolor:1, Iris-virginica:2
le = LabelEncoder()
label = le.fit_transform(df['species'])
df.drop("species", axis=1, inplace=True)
df["species"] = label

x=df.drop('species',axis=1)
y=df['species']
x_train, x_test, y_train, y_test = train_test_split(x,y ,random_state=0,
                                   test_size=0.3,
                                   shuffle=True)

classifier = RandomForestClassifier()
classifier.fit(x_train, y_train)

# predicting on the test dataset
y_pred = classifier.predict(x_test)

# finding out the accuracy
from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, y_pred)

import pickle
pickle_out = open("classifier.pkl", "wb")
pickle.dump(classifier, pickle_out)
pickle_out.close()
