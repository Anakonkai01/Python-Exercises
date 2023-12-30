import sympy as sp
import matplotlib.pyplot as plt
import numpy as np

x = sp.symbols('x')
f = sp.E**(-1/2*x**2)
f_func = sp.lambdify(x, f, 'numpy')
x_values = np.linspace(-5, 5, 1000)
f_values = f_func(x_values)
plt.figure(figsize=(8, 6))
plt.plot(x_values, f_values, label='e**(-1/2*x**2)')
plt.fill_between(x_values, f_values, color='lightcoral', alpha=0.4, label='Area under the curve')
plt.title('Function: e^(1/2 * x**2) over [-4, 4] with Area under the curve')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

area_within_interval = sp.integrate(f, (x, -sp.oo, sp.oo))
print(f"Area under the curve of e**(-1/2*x**2) over [-4, 4]: {area_within_interval.evalf()}")
