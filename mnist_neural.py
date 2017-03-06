import pandas as pd
import numpy as np
#from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

train_df = pd.read_csv("D:/Vacation/ML/Datasets/MNIST/mnist_train.csv", header=None)
test_df = pd.read_csv("D:/Vacation/ML/Datasets/MNIST/mnist_test.csv", header=None)

n_train, n_features = train_df.shape
n_test = test_df.shape[0]

n_features -=1

X_train = np.zeros(shape=(n_train, n_features))
X_test = np.zeros(shape=(n_test, n_features))
y_train = np.zeros(shape=(n_train, 1))
y_test = np.zeros(shape=(n_train, 1))

X_train = train_df.iloc[0:n_train, 1:n_features+1].values
X_test = test_df.iloc[0:n_test, 1:n_features+1].values
y_train = train_df.iloc[0:n_train, [0]].values
y_test = test_df.iloc[0:n_test, [0]].values

#Data preprocessing
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#Training the data with Multi-layer Perceptron Classifier
mlp = MLPClassifier(hidden_layer_sizes=(400, 400))
mlp.fit(X_train, y_train)

#Calculate aaccuracy
predicted = mlp.predict(X_test)
print(accuracy_score(y_test, predicted))