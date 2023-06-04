import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange, CubicSpline

# Визначення функції f(x)
def f(x):
    if -1 <= x <= 0:
        return (x - 0.5)**2 + 0.75
    elif 0 <= x <= 2:
        return 1 - 4 * (x - 0.5)**2
    elif 2 <= x <= 4:
        return x**2 - 2 * x + 1

# Задані інтервали [a, b]
a = -1
b = 4

# Кількість вузлів для побудови інтерполяційного полінома
n = 10

# Рівновіддалені вузли
x_equidistant = np.linspace(a, b, n)
y_equidistant = np.array([f(x) for x in x_equidistant])

# Чебишовські вузли
x_chebyshev = np.cos((2 * np.arange(1, n + 1) - 1) * np.pi / (2 * n)) * (b - a) / 2 + (a + b) / 2
y_chebyshev = np.array([f(x) for x in x_chebyshev])

# Побудова інтерполяційних поліномів
poly_equidistant = lagrange(x_equidistant, y_equidistant)
poly_chebyshev = lagrange(x_chebyshev, y_chebyshev)

# Графік 1: f(x), (P^E)n(x), (P^T)n(x)
x = np.linspace(a, b, 500)
y = np.array([f(val) for val in x])

plt.figure(figsize=(10, 6))
plt.plot(x, y, label='f(x)')
plt.plot(x, poly_equidistant(x), label='P^E_n(x)')
plt.plot(x, poly_chebyshev(x), label='P^T_n(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Interpolation Polynomials')
plt.grid(True)
plt.show()

# Графік 2: f(x) - (P^E)n(x), f(x) - (P^T)n(x)
plt.figure(figsize=(10, 6))
plt.plot(x, y - poly_equidistant(x), label='f(x) - P^E_n(x)')
plt.plot(x, y - poly_chebyshev(x), label='f(x) - P^T_n(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Interpolation Errors')
plt.grid(True)
plt.show()

# Побудова кубічного природнього сплайна
nodes = np.linspace(a, b, n + 2)
values = np.array([f(val) for val in nodes])
natural_spline = CubicSpline(nodes, values, bc_type='natural')

# Графік 1: f(x), s(x)
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='f(x)')
plt.plot(x, natural_spline(x), label='s(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Cubic Natural Spline')
plt.grid(True)
plt.show()

# Графік 2: f(x) - s(x)
plt.figure(figsize=(10, 6))
plt.plot(x, y - natural_spline(x), label='f(x) - s(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Interpolation Error (Cubic Natural Spline)')
plt.grid(True)
plt.show()
