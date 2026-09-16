import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# 1. TẠO DỮ LIỆU

np.random.seed(42)

X = np.linspace(0, 10, 50)
y = np.sin(X) + np.random.normal(0, 0.3, 50)

X = X.reshape(-1, 1)


# 2. CHIA TRAINING VÀ TEST

X_train = X[:30]
y_train = y[:30]

X_test = X[30:]
y_test = y[30:]


# 3. TẠO MÔ HÌNH BẬC CAO

poly = PolynomialFeatures(degree=15)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

model = LinearRegression()

model.fit(X_train_poly, y_train)


# 4. KIỂM TRA OVERFITTING

train_pred = model.predict(X_train_poly)
test_pred = model.predict(X_test_poly)

train_error = mean_squared_error(y_train, train_pred)
test_error = mean_squared_error(y_test, test_pred)

print("Train error:", train_error)
print("Test error:", test_error)


# 5. K-FOLD CROSS VALIDATION

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

degrees = [1, 2, 3, 4, 5, 10, 15]

errors = []

for degree in degrees:

    fold_errors = []

    for train_index, val_index in kf.split(X_train):

        X1 = X_train[train_index]
        X2 = X_train[val_index]

        y1 = y_train[train_index]
        y2 = y_train[val_index]

        poly = PolynomialFeatures(degree)

        X1_poly = poly.fit_transform(X1)
        X2_poly = poly.transform(X2)

        model = LinearRegression()

        model.fit(X1_poly, y1)

        prediction = model.predict(X2_poly)

        error = mean_squared_error(y2, prediction)

        fold_errors.append(error)

    errors.append(np.mean(fold_errors))


# 6. CHỌN BẬC TỐT NHẤT

best_degree = degrees[np.argmin(errors)]

print("Bậc tốt nhất:", best_degree)
print("CV error nhỏ nhất:", min(errors))


# 7. VẼ KẾT QUẢ

plt.plot(degrees, errors, marker="o")

plt.xlabel("Bậc đa thức")
plt.ylabel("CV Error")
plt.title("K-Fold Cross Validation")

plt.show()