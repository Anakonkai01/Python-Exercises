import numpy as np
from scipy.optimize import leastsq
import matplotlib.pyplot as plt

# Define the data points
data_points = np.array([(1, 7.9), (2, 5.4), (3, -9)])

# Extract x and y values
x_values = data_points[:, 0]
y_values = data_points[:, 1]

# Define the model function
def model(params, x):
    A, B = params
    return A * np.cos(x) + B * np.sin(x)

# Define the error function
def error(params, x, y):
    return y - model(params, x)

# Initial guess for the parameters A and B
initial_guess = [1, 1]

# Perform least squares optimization
result = leastsq(error, initial_guess, args=(x_values, y_values))

# Extract the optimized parameters
A_opt, B_opt = result[0]

# Print the optimized parameters
print(f"The optimized parameters are A = {A_opt:.6f} and B = {B_opt:.6f}")

# Define a function to calculate y using the optimized parameters
def calculate_y(x):
    return A_opt * np.cos(x) + B_opt * np.sin(x)

# Calculate y values for the x values using the optimized parameters
y_fitted = calculate_y(x_values)

# Print the fitted y values
print(f"The fitted y values are: {y_fitted}")

# Plot the original data points and the best-fit curve
x_range = np.linspace(min(x_values), max(x_values), 100)
y_range = calculate_y(x_range)

plt.scatter(x_values, y_values, color='red', label='Original Data')
plt.plot(x_range, y_range, label='Best Fit Curve')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Least Squares Fit: y = Acos(x) + Bsin(x)')
plt.legend()
plt.grid(True)
plt.show()
