import numpy as np

# Trọng số
w = np.array([1, 2, -10])

# Điểm dữ liệu có bias
x = np.array([3, 4, 1])

# Nhãn thực tế
y = -1

# 1. Tính w^T x
z = np.dot(w, x)

print("w^T x =", z)

# 2. Tính nhãn dự đoán
if z >= 0:
    y_pred = 1
else:
    y_pred = -1

print("Nhãn dự đoán =", y_pred)

# 3. Kiểm tra phân lớp sai
if y_pred != y:
    print("Điểm dữ liệu bị phân lớp sai")
else:
    print("Điểm dữ liệu được phân lớp đúng")