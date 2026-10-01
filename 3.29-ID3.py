# BÀI 3.29 - XÂY DỰNG LỚP PERCEPTRON

import numpy as np

class Perceptron:

    def __init__(self, learning_rate=0.1, epochs=10):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None

    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])

        for epoch in range(self.epochs):

            for i in range(len(X)):

                prediction = self.predict(X[i])

                if prediction != y[i]:
                    self.weights = self.weights + self.learning_rate * y[i] * X[i]

    def predict(self, x):
        value = np.dot(self.weights, x)

        if value >= 0:
            return 1
        else:
            return -1


X = np.array([
    [2, 3],
    [1, 1],
    [-2, -1],
    [-3, -2]
])

y = np.array([1, 1, -1, -1])

model = Perceptron(
    learning_rate=0.1,
    epochs=10
)

model.fit(X, y)

print("Trọng số:", model.weights)

x_new = np.array([2, 2])

prediction = model.predict(x_new)

print("Nhãn dự đoán:", prediction)