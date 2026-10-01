# BÀI 3.26 - GRADIENT DESCENT

def f(x):
    return x**2 - 4*x + 5

def df(x):
    return 2*x - 4

x = 5
eta = 0.2

print("Bước 0:")
print("x =", x)
print("f(x) =", f(x))

for i in range(1, 5):
    x = x - eta * df(x)

    print("\nBước", i)
    print("x =", x)
    print("f(x) =", f(x))