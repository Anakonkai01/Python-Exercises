import numpy as np
from sympy import sympify, lambdify

def is_one_to_one(func, domain):
    seen_pairs = set()

    for input_val in domain:
        output_val = func(input_val)
        pair = (input_val, output_val)

        if pair in seen_pairs:
            # If the same (input, output) pair is encountered again, return False
            return False

        seen_pairs.add(pair)

    return True

def f1(x):
    return x**3 - x/2

# Convert the function to a lambda function
f1_lambda = lambdify('x', f1, 'numpy')

# Increase granularity by using a smaller step size
domain1 = np.arange(-100.0, 101.0, 0.01)
print("Function f(x) = x^3 - x/2 is one-to-one:", is_one_to_one(f1_lambda, domain1))

def f2(x):
    return x**2 + x/2

# Convert the function to a lambda function
f2_lambda = lambdify('x', f2, 'numpy')

# Increase granularity by using a smaller step size
domain2 = np.arange(-100.0, 101.0, 0.01)
print("Function f(x) = x^2 + x/2 is one-to-one:", is_one_to_one(f2_lambda, domain2))
