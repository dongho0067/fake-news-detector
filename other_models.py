import pandas as pd
import numpy as np # linear algebra
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import data_processor as dp
from sklearn import metrics
import seaborn as sns


#Can implement Logistic Regression and Random Forrest really easily
#The focus of the program is the LSTM, so using these 2 would serve as a good comparison to see if our LSTM is better.
#Try to implement this with new data that we have to clean and process!
features=dp.data['title']+ " "+ dp.data['text']
X=features.values
y=dp.data['label']

vectorizer= TfidfVectorizer()
X=vectorizer.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2, stratify=y, random_state=42)

model=LogisticRegression().fit(X_train, Y_train)
test=model.predict(X_test)
print(accuracy_score(test, Y_test))

model = RandomForestClassifier()
model.fit(X_train, Y_train)

test=model.predict(X_test)
print("This is Random Forrest", accuracy_score(test, Y_test))

#We want our LSTM to do at least as good as these 2

#Need to Create Plots and Confusion Matrices for these 2 methods.


