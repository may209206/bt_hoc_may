def f(x):
    return x**2 - 4*x + 5
def df(x):
    return 2*x - 4
x = 5
eta = 0.2
print("x(0) =", x)
print("f(x(0)) =", f(x))
for i in range(4):
    gradient = df(x)
    print(f"\nBước {i + 1}")
    print("x =", x)
    print("f'(x) =", gradient)
    x_new = x - eta * gradient
    print("x mới =", x_new)
    print("f(x mới) =", f(x_new))
    x = x_new
print("\nKết quả cuối:")
print("x =", x)
print("f(x) =", f(x))