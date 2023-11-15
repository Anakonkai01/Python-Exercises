import numpy as np
from sympy import symbols, diff, solve

def is_one_to_one(func, domain):
    x = symbols('x')
    derivative = diff(func, x)

    for input_val in domain:
        output_val = func.subs(x, input_val)
        derivative_val = derivative.subs(x, input_val)

        if derivative_val == 0:
            # If the derivative is zero, check if there are multiple values for x
            solutions = solve(derivative, x)
            if len(solutions) > 1:
                return False

    return True

def f1(x):
    return x**3 - x/2

domain1 = np.arange(-100.0, 101.0, 0.01)
print("Function f(x) = x^3 - x/2 is one-to-one:", is_one_to_one(f1, domain1))

def f2(x):
    return x**2 + x/2

domain2 = np.arange(-100.0, 101.0, 0.01)
print("Function f(x) = x^2 + x/2 is one-to-one:", is_one_to_one(f2, domain2))
