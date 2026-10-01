# BÀI 3.30 - PHÂN LỚP NHỊ PHÂN BẰNG PERCEPTRON

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


X = np.array([
    [100, 1],
    [150, 1],
    [200, 2],
    [250, 2],
    [300, 3],
    [350, 3],
    [400, 4],
    [450, 4],
    [500, 5],
    [550, 5]
])

y = np.array([
    -1,
    -1,
    -1,
    -1,
    -1,
    1,
    1,
    1,
    1,
    1
])


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)


class Perceptron:

    def __init__(self, learning_rate=0.01, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None

    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])

        for epoch in range(self.epochs):

            for i in range(len(X)):

                value = np.dot(self.weights, X[i])

                if y[i] * value <= 0:
                    self.weights = (
                        self.weights
                        + self.learning_rate * y[i] * X[i]
                    )

    def predict(self, X):
        result = []

        for x in X:
            value = np.dot(self.weights, x)

            if value >= 0:
                result.append(1)
            else:
                result.append(-1)

        return np.array(result)


model = Perceptron(
    learning_rate=0.01,
    epochs=100
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label=1)
recall = recall_score(y_test, y_pred, pos_label=1)
f1 = f1_score(y_test, y_pred, pos_label=1)


print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)