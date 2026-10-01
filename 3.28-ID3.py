# BÀI 3.28 - PERCEPTRON

import numpy as np

w = np.array([-2, 1, 0])
x = np.array([2, 3, 1])
y = 1

wx = np.dot(w, x)

print("w^T x ban đầu =", wx)

if y * wx <= 0:
    print("Mẫu bị phân lớp sai")

    w = w + y * x

    print("Trọng số sau cập nhật =", w)

    wx_new = np.dot(w, x)

    print("w^T x sau cập nhật =", wx_new)

else:
    print("Mẫu được phân lớp đúng")