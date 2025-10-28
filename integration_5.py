import numpy as np

def function_1(x, y):  
    return np.sin(x**2 + y**2) / (1 + x + y)

def grid_1(a, b, c, d, n):
    return np.meshgrid(np.linspace(a, b, n), np.linspace(c, d, n))

def get_double_simpson_integral(func, X, Y, a, b, c, d, n_x, n_y):  
    Z = func(X, Y) 
    weights_x = np.ones(n_x)
    weights_x[1:-1:2], weights_x[2:-2:2] = 4, 2 
    weights_y = np.ones(n_y)
    weights_y[1:-1:2], weights_y[2:-2:2] = 4, 2 
    h_x, h_y = (b - a) / (n_x - 1), (d - c) / (n_y - 1)  
    return np.dot(weights_y, np.dot(Z, weights_x)) * (h_x * h_y) / 9

def calc_integral_2(func, grid, a, b, c, d, eps):
    n = 3 
    X1, Y1 = grid(a, b, c, d, n)
    I_prev = get_double_simpson_integral(func, X1, Y1, a, b, c, d, n, n)
    n *= 2
    if n % 2 == 0: n += 1 
    X2, Y2 = grid(a, b, c, d, n)
    I_curr = get_double_simpson_integral(func, X2, Y2, a, b, c, d, n, n)
    while abs(I_curr - I_prev) > (1 / 16) * eps:
        print(f"I_n = {I_prev}; I_2n = {I_curr}; R = {abs(I_prev - I_curr)}")
        I_prev = I_curr
        n = n * 2 - 1  
        X, Y = grid(a, b, c, d, n)
        I_curr = get_double_simpson_integral(func, X, Y, a, b, c, d, n, n)
    return I_curr

print("f(x,y) = sin(x**2 + y**2)/(1+x+y) по области [0,1]; [1;2]")
print("Вычисленное значение = ", calc_integral_2(function_1, grid_1, 0, 1, 1, 2, 10**(-15)))
