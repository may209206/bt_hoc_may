import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.1, epochs=10):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w = None
        self.b = 0

    def fit(self, X, y):
        self.w = np.zeros(X.shape[1])

        for _ in range(self.epochs):
            for i in range(len(X)):
                z = np.dot(X[i], self.w) + self.b

                if z >= 0:
                    y_pred = 1
                else:
                    y_pred = -1

                if y_pred != y[i]:
                    self.w = self.w + self.learning_rate * y[i] * X[i]
                    self.b = self.b + self.learning_rate * y[i]

    def predict(self, X):
        z = np.dot(X, self.w) + self.b

        if z.ndim == 0:
            return 1 if z >= 0 else -1

        return np.where(z >= 0, 1, -1)


# Dữ liệu
X = np.array([
    [1, 1],
    [2, 2],
    [-1, -1],
    [-2, -2]
])

y = np.array([1, 1, -1, -1])

# Tạo mô hình
model = Perceptron(learning_rate=0.1, epochs=10)

# Huấn luyện
model.fit(X, y)

# Dữ liệu mới
X_new = np.array([
    [3, 3],
    [-3, -3]
])

# Dự đoán
print("w =", model.w)
print("b =", model.b)
print("Dự đoán:", model.predict(X_new))