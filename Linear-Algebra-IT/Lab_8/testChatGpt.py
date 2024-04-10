import numpy as np
import matplotlib.pyplot as plt

def least_squares_cos_sin(data):
    x = data[:, 0]
    y = data[:, 1]
    
    A = np.column_stack((np.cos(x), np.sin(x)))
    coeffs = np.linalg.lstsq(A, y, rcond=None)[0]
    
    # Calculate predicted y values
    y_predicted = np.dot(A, coeffs)
    
    # Plot the data points and the line of best fit
    plt.scatter(x, y, label='Data Points')
    plt.plot(x, y_predicted, color='red', label='Line of Best Fit')
    
    # Add labels and legend
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Least Squares Fit: y = Acos(x) + Bsin(x)')
    plt.legend()
    
    # Show plot
    plt.grid(True)
    plt.show()
    
    return coeffs

# Given data points
data = np.array([[1, 7.9], [2, 5.4], [3, -9]])

# Perform least squares regression and plot
coefficients = least_squares_cos_sin(data)
print("Coefficients (A, B):", coefficients)
