import pickle

import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


data_frame = pd.read_csv("data.csv")

X = data_frame[["radius_mean", "texture_mean", "perimeter_mean", "area_mean"]]
Y = data_frame['diagnosis']

# model training

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
model = LogisticRegression()
model.fit(X_train, Y_train)
pickle.dump(model, open("model.pkl","wb"))

print(data_frame.head())

# accuracy on training data
# X_train_prediction = model.predict(X_train)
# training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
# print('Accuracy on training data = ', training_data_accuracy)

# accuracy on test data
# X_test_prediction = model.predict(X_test)
# test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
# print('Accuracy on testing data = ', test_data_accuracy)

