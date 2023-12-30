import numpy as np
import matplotlib.pyplot as plt

def fibonacci_search(func, a, b, epsilon=1e-6):
    # Fibonacci sequence generation
    fib_sequence = [1, 1]
    while fib_sequence[-1] < (b - a) / epsilon:
        fib_sequence.append(fib_sequence[-1] + fib_sequence[-2])

    # Initial points
    x1 = a + (b - a) * fib_sequence[-3] / fib_sequence[-1]
    x2 = a + (b - a) * fib_sequence[-2] / fib_sequence[-1]

    # Initial function evaluations
    f_x1 = func(x1)
    f_x2 = func(x2)

    # Iteration counter
    iteration = 1

    # Store iteration results for plotting/table
    results = []

    while abs(b - a) > epsilon:
        results.append([iteration, a, b, x1, x2, f_x1, f_x2])

        if f_x1 < f_x2:
            b = x2
            x2 = x1
            x1 = a + (b - a) * fib_sequence[-3] / fib_sequence[-1]
            f_x2 = f_x1
            f_x1 = func(x1)
        else:
            a = x1
            x1 = x2
            x2 = a + (b - a) * fib_sequence[-2] / fib_sequence[-1]
            f_x1 = f_x2
            f_x2 = func(x2)

        iteration += 1

    # Add the final iteration
    results.append([iteration, a, b, x1, x2, f_x1, f_x2])

    return results

# Objective function
def objective_function(x):
    return x**2

# Interval and tolerance
a = -2
b = 1
epsilon = 0.3

# Apply Fibonacci Search
results = fibonacci_search(objective_function, a, b, epsilon)

# Display the results in a table
print("Iteration |   a   |   b   |   x1   |   x2   |  f(x1)  |  f(x2) ")
print("="*66)
for row in results:
    print("{:9} | {:5.2f} | {:5.2f} | {:6.2f} | {:6.2f} | {:7.2f} | {:7.2f}".format(*row))

# Plot the function and the iteration points
x_values = np.linspace(a, b, 100)
y_values = objective_function(x_values)

plt.plot(x_values, y_values, label="f(x) = x^2")
plt.scatter([row[3] for row in results], [row[5] for row in results], color='red', label='Iteration Points', zorder=5)
plt.title("Fibonacci Search for Minimum of f(x) = x^2")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(True)
plt.show()
