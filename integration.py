import numpy as np
import sympy as sp
from sympy import symbols, integrate, sympify
from sympy.parsing.sympy_parser import standard_transformations, function_exponentiation, parse_expr
from sympy.utilities.lambdify import lambdify
from scipy.integrate import trapezoid, simpson, fixed_quad
from scipy.special import roots_legendre
from tabulate import tabulate
import matplotlib.pyplot as plt
from math import factorial, ceil
 
def sympy_integral(func_str, a, b):
    x = symbols('x')
    return float(sp.N(integrate(sympify(func_str.replace('np.', '')), (x, a, b))))
 
def get_max_derivative(function_str, n_dir, a, b):
    transformations = standard_transformations + (function_exponentiation,)
    cleaned_string = function_str.replace('np.', '')
    x = sp.Symbol('x')
    expr = parse_expr(cleaned_string, transformations=transformations)
    nth_deriv_exact = expr.diff(x, n_dir) 
    nth_deriv_func = lambdify(x, nth_deriv_exact, 'numpy')
    x_values = np.linspace(a, b, 1000)
    deriv_values = np.abs(nth_deriv_func(x_values))
    return deriv_values.max()
 
class Integrator:
    def __init__(self, function_str, a, b, n):
        self.function_str = function_str
        self.calc_function = eval(f"lambda x: {function_str}")
        self.a, self.b, self.n, self.h = a, b, n, (b - a) / (n - 1)
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!! h = {self.h} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        self.x = np.linspace(self.a, self.b, self.n)
        self.y = self.calc_function(self.x)
 
    def get_trapecia_integral(self):
        integral = ((self.y[0] + self.y[-1]) / 2 + np.sum(self.y[1:-1])) * (self.h)
        R = -1.0 * ((self.b - self.a) * (self.h ** 2) / 12) * get_max_derivative(self.function_str, 2, self.a, self.b)
        return integral, R, trapezoid(self.y, self.x)
  
    def get_trapecia_with_optimal_h(self, eps, sposob):
        if sposob == "first":
            self.h = (12 * eps / (self.b - self.a) / get_max_derivative(self.function_str, 2, self.a, self.b)) ** (1 / 2)
            self.n = int(ceil((self.b - self.a) / self.h)) + 1
            self.h = (self.b - self.a) / (self.n - 1)
            self.x = np.linspace(self.a, self.b, self.n)
            self.y = self.calc_function(self.x)
            I, R, I_s = self.get_trapecia_integral()
            return self.h, self.n, I, R, I_s
        elif sposob == "second":
            C = 1 / 3
            n_1 = 2
            n_2 = 4 
            self.h = (self.b - self.a) / (n_1)
            self.x = np.linspace(self.a, self.b, int(n_1 + 1))
            self.y = self.calc_function(self.x)
            I_n, R_n, I_ns = self.get_trapecia_integral()
            self.h = (self.b - self.a) / (n_2)
            self.x = np.linspace(self.a, self.b, int(n_2 + 1))
            self.y = self.calc_function(self.x)
            I_2n, R_2n, I_2ns = self.get_trapecia_integral()
            print("----------------------- Формула трапеций с подбором шага h (2 способ) -----------------------------")
            print(f"I_n = {I_n}; I_2n = {I_2n}; Разница = {abs(I_2n - I_n)}")
            while abs(I_2n - I_n) > C * eps:
                n_1 = n_2
                I_n = I_2n
                n_2 *= 2
                self.h = (self.b - self.a) / (n_2)
                self.x = np.linspace(self.a, self.b, n_2+1)
                self.y = self.calc_function(self.x)
                I_2n, R_2n, I_2ns = self.get_trapecia_integral()
                print(f"I_n = {I_n}; I_2n = {I_2n}; Разница = {abs(I_2n - I_n)}")
            return I_2n
 
    def get_simpson_integral(self):
        if self.n % 2 == 0:
            raise ValueError("n должно быть нечетным")
        integral = (self.y[0] + self.y[-1] + 4 * np.sum(self.y[1:-1:2]) + 2 * np.sum(self.y[2:-2:2])) * (self.h / 3)
        R = -1.0 * ((self.b - self.a) * (self.h ** 4) / 180) * get_max_derivative(self.function_str, 4, self.a, self.b)
        return integral, R, simpson(self.y, self.x)
  
    def get_simpson_with_optimal_h(self, eps, sposob):
        if sposob == "first":
            self.h = (180 * eps / (self.b - self.a) / get_max_derivative(self.function_str, 4, self.a, self.b)) ** (1 / 4)
            self.n = int(ceil((self.b - self.a) / self.h)) + 1
            if self.n % 2 == 0:
                self.n += 1
            self.h = (self.b - self.a) / (self.n - 1)
            self.x = np.linspace(self.a, self.b, self.n)
            self.y = self.calc_function(self.x)
            I, R, I_s = self.get_simpson_integral()
            return self.h, self.n, I, R, I_s
        elif sposob == "second":
            C = 1 / 16
            n_1 = 2
            n_2 = 4 
            self.h = (self.b - self.a) / (n_1)
            self.x = np.linspace(self.a, self.b, int(n_1 + 1))
            self.y = self.calc_function(self.x)
            I_n, R_n, I_ns = self.get_simpson_integral()
            self.h = (self.b - self.a) / (n_2)
            self.x = np.linspace(self.a, self.b, int(n_2 + 1))
            self.y = self.calc_function(self.x)
            I_2n, R_2n, I_2ns = self.get_simpson_integral()
            print("----------------------- Формула Симпсона с подбором шага h (2 способ) -----------------------------")
            print(f"I_n = {I_n}; I_2n = {I_2n}; Разница = {abs(I_2n - I_n)}")
            while abs(I_2n - I_n) > C * eps:
                n_1 = n_2
                I_n = I_2n
                n_2 *= 2
                self.h = (self.b - self.a) / (n_2)
                self.x = np.linspace(self.a, self.b, n_2+1)
                self.y = self.calc_function(self.x)
                I_2n, R_2n, I_2ns = self.get_simpson_integral()
                print(f"I_n = {I_n}; I_2n = {I_2n}; Разница = {abs(I_2n - I_n)}")
            return I_2n
 
    def get_3_8_integral(self):
        if (self.n - 1) % 3 != 0:
            raise Exception("n - 1 должно быть кратно трем")
        integral = (self.y[0] + self.y[-1] + 3 * np.sum(self.y[1:-1:3]) + 3 * np.sum(self.y[2:-1:3]) + 2 * np.sum(self.y[3:-1:3])) * (3 * self.h / 8)
        R = -1.0 * ((self.b - self.a) * (self.h ** 4) / 80) * get_max_derivative(self.function_str, 4, self.a, self.b)
        return integral, R, simpson(self.y, self.x) 
 
    def rectangle_integral(self, flag):
        if flag == 'left':
            integral = np.sum(self.y[:-1]) * self.h
            R = -1.0 * ((self.b - self.a) * self.h / 2) * get_max_derivative(self.function_str, 1, self.a, self.b)
        elif flag == 'right':
            integral = np.sum(self.y[1:]) * self.h 
            R = -1.0 * ((self.b - self.a) * self.h / 2) * get_max_derivative(self.function_str, 1, self.a, self.b)
        elif flag == 'center':
            x_center = (self.x[:-1] + self.x[1:]) / 2
            y_center = self.calc_function(x_center) 
            integral = np.sum(y_center) * self.h
            R = -1.0 * ((self.b - self.a) * (self.h ** 2) / 24) * get_max_derivative(self.function_str, 2, self.a, self.b)
        return integral, R
 
    def get_gauss_integral(self, n):
        T_i, A_i = roots_legendre(n)
        X_i = (self.b + self.a) / 2 + ((self.b - self.a) / 2) * T_i
        Y_i = self.calc_function(X_i)
        integral = ((self.b - self.a) / 2) * np.sum(np.array([A_i[i] * Y_i[i] for i in range(len(Y_i))]))
        return integral, fixed_quad(self.calc_function, self.a, self.b, n=n)[0]
  
 
'''a = float(input("Введите a: "))
b = float(input("Введите b: "))
n = int(input("Введите количество узлов интегрирования: "))
func_str = input("Введите функцию: ")'''
 
a = 0
b = 5
n = 270001
n_gauss = 5000
func_str = "np.cos(5*x*x)"


 
try:
    result = sympy_integral(func_str, a, b)
    print(f"∫({func_str})dx от {a} до {b} = {result}")
except:
    print("Не удалось вычислить точное значение интеграла")
    result = None

print("----------------------- Задание 1 -----------------------")
integrator = Integrator(func_str, a, b, n)
 
headers = ["Метод интегрирования", "Численное значение", "R", "Проверка SciPy"]
table_data = []
 
try:
    trapecia_integral, R_trapecia, trapecia_integral_scipy  = integrator.get_trapecia_integral()
    table_data.append(["Формула трапеций", trapecia_integral, R_trapecia, trapecia_integral_scipy])
except Exception as e:
    table_data.append(["Формула трапеций", "Ошибка", str(e)])
 
try:
    simpson_integral, R_simpson, simpson_integral_scipy = integrator.get_simpson_integral()
    table_data.append(["Формула Симпсона", simpson_integral, R_simpson, simpson_integral_scipy])
except Exception as e:
    table_data.append(["Формула Симпсона", "Ошибка", str(e)])
 
try:
    three_eight_integral, R_three_eight, three_eight_integral_scipy = integrator.get_3_8_integral()
    table_data.append(["Формула 3/8", three_eight_integral, R_three_eight, three_eight_integral_scipy])
except Exception as e:
    table_data.append(["Формула 3/8", "Ошибка", str(e)])
 
try:
    left_rectangle_integral, R_left_rectangle = integrator.rectangle_integral("left")
    table_data.append(["Формула левых прямоугольников", left_rectangle_integral, R_left_rectangle])
except Exception as e:
    table_data.append(["Формула левых прямоугольников", "Ошибка", str(e)])
 
try:
    right_rectangle_integral, R_right_rectangle = integrator.rectangle_integral("right")
    table_data.append(["Формула правых прямоугольников", right_rectangle_integral, R_right_rectangle])
except Exception as e:
    table_data.append(["Формула правых прямоугольников", "Ошибка", str(e)])
 
try:
    center_rectangle_integral, R_center_rectangle = integrator.rectangle_integral("center")
    table_data.append(["Формула центральных прямоугольников", center_rectangle_integral, R_center_rectangle])
except Exception as e:
    table_data.append(["Формула центральных прямоугольников", "Ошибка", str(e)])
 
try:
    gauss_integral, gauss_integral_scipy = integrator.get_gauss_integral(n_gauss)
    table_data.append([f"Гаусс ({n_gauss} степень)", gauss_integral, "", gauss_integral_scipy])
except Exception as e:
    table_data.append([f"Гаусс ({n_gauss} степень)", "Ошибка", str(e)])
 
print(tabulate(table_data, headers=headers, tablefmt="grid", floatfmt=("", ".15f","",".15f")))

print("----------------------- Задание 2 -----------------------")
headers_2 = ["Метод интегрирования", "h", "n", "Численное значение", "R", "Проверка SciPy"]
table_data_2 = []
print("Способ номер 1")
try:
    trapecia_h_opt_1, trapecia_n_opt_1, trapecia_I_opt_1, trapecia_R_opt_1, trapecia_I_s_optimal_1 = integrator.get_trapecia_with_optimal_h(10 ** (-12), "first")
    table_data_2.append(["Формула трапеций (способ 1)", trapecia_h_opt_1, trapecia_n_opt_1, trapecia_I_opt_1, trapecia_R_opt_1, trapecia_I_s_optimal_1])
except Exception as e:
    table_data_2.append(["Формула трапеций (способ 1)", "Ошибка", str(e)])

try:
    simpson_h_opt_1, simpson_n_opt_1, simpson_I_opt_1, simpson_R_opt_1, simpson_I_s_optimal_1 = integrator.get_simpson_with_optimal_h(10 ** (-12), "first")
    table_data_2.append(["Формула Симпсона (способ 1)", simpson_h_opt_1, simpson_n_opt_1, simpson_I_opt_1, simpson_R_opt_1, simpson_I_s_optimal_1])
except Exception as e:
    table_data_2.append(["Формула Симпсона (способ 1)", "Ошибка", str(e)])

print(tabulate(table_data_2, headers=headers_2, tablefmt="grid", floatfmt=("", ".15f", "",".15f","",".15f")))

I_2n = integrator.get_trapecia_with_optimal_h(10**(-12), "second")
print("Погрешность = ", 10**(-12))
print("Вычисленное значение = ", I_2n)

I_2n = integrator.get_simpson_with_optimal_h(10**(-12), "second")
print("Погрешность = ", 10**(-12))
print("Вычисленное значение = ", I_2n)
