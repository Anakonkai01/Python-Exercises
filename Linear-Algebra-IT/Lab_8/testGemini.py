import numpy as np
from scipy.optimize import least_squares

def objective_function(params, x, y):
  """
  Objective function for least squares fit of y = A * cos(x) + B * sin(x).

  Args:
      params: A tuple containing the parameters A and B.
      x: A list of x-coordinate values.
      y: A list of y-coordinate values.

  Returns:
      A list of residuals (differences between predicted and actual y-values).
  """
  A, B = params
  model = A * np.cos(x) + B * np.sin(x)
  return y - model

# Sample data points
x = [1, 2, 3]
y = [7.9, 5.4, -9]

# Initial guess for A and B (adjust as needed)
initial_guess = [1.0, 1.0]

# Perform least squares fit using least_squares
result = least_squares(objective_function, initial_guess, args=(x, y))

# Extract fitted parameters
A, B = result.x

# Print the fitted model
print("Fitted model: y = {:.4f} * cos(x) + {:.4f} * sin(x)".format(A, B))
