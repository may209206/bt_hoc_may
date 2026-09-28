import numpy as np

# Trọng số ban đầu
w = np.array([-2, 1, 0])

# Điểm dữ liệu có bias
x = np.array([2, 3, 1])

# Nhãn thực tế
y = 1

# Tính w^T x
z = np.dot(w, x)

print("w^T x =", z)

# Dự đoán nhãn
if z >= 0:
    y_pred = 1
else:
    y_pred = -1

print("Nhãn dự đoán =", y_pred)

# Kiểm tra phân lớp sai
if y_pred != y:
    print("Mẫu bị phân lớp sai")

    # Cập nhật Perceptron
    w = w + y * x

    print("w sau cập nhật =", w)

    # Tính lại w^T x
    z_new = np.dot(w, x)

    print("w^T x sau cập nhật =", z_new)

else:
    print("Mẫu được phân lớp đúng")