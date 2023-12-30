import sympy as sp
import matplotlib.pyplot as plt
import numpy as np

x = sp.symbols('x')

f = x**2 * sp.cos(x)

interval = (-4, 9)

f_func = sp.lambdify(x, f, 'numpy')

x_values = np.linspace(interval[0], interval[1], 1000)
f_values = f_func(x_values)

plt.figure(figsize=(8, 6))
plt.plot(x_values, f_values, label='x**2*cos(x)')
plt.fill_between(x_values, f_values, color='red', alpha=0.4, label='Area under the curve')
plt.title('Function: x**2*cos(x) over [-4, 9] with Area under the curve')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

area = sp.integrate(f, (x, interval[0], interval[1]))
print(f"Area under the curve of x**2*cos(x) over [-4, 9]: {area.evalf()}")
