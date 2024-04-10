import numpy as np
import matplotlib.pyplot as plt

def exercise1():
    print("Exercise 1:")    
    A = np.array([[2, 2], [2, 3]])
    b = np.array([[4], [4]])

    x = np.linalg.inv(A.T @ A) @ A.T @ b

    print(f"The least square solution to Ax = b is:\n{x}")

def exercise2():
    print()
    print()
    print("Exercise 2:")

    A = np.array([0, 0, 1, 0, 1, 1, 1, 2, 1, 1, 0, 1, 4, 1, 1, 4, 2, 1]).reshape(6, 3)
    b = np.array([0.5, 1.6, 2.8, 0.8, 5.1, 5.9])

    x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

    print(f"The least squares solution is:\n"
        f"c = {x[0]}\n"
        f"d = {x[1]}\n"
        f"e = {x[2]}")
    
def exercise3():
    print()
    print()
    print("Exercise 3:")
    def least_squares_line(data):
        x = np.array([point[0] for point in data])
        y = np.array([point[1] for point in data])
        a1 = np.sum((x - np.mean(x)) * (y - np.mean(y))) / np.sum(np.power(x - np.mean(x), 2))
        a0 = np.mean(y) - a1 * np.mean(x)
        return a0, a1

    data_sets = {
        'a': [(0, 1), (1, 1), (2, 2), (3, 2)],
        'b': [(1, 0), (2, 1), (4, 2), (5, 3)],
        'c': [(-1, 0), (0, 1), (1, 2), (2, 4)],
        'd': [(2, 3), (3, 2), (5, 1), (6, 0)]
    }

    for name, data in data_sets.items():
        a0, a1 = least_squares_line(data)
        print(f"Data set {name}: y = {a0:.4f} + {a1:.4f}x")

def exercise4():
    print()
    print()
    print("Exercise 4:")
    def least_squares_line(mileage, friction_index):
        x = np.array(mileage)
        y = np.array(friction_index)
        a1 = np.sum((x - np.mean(x)) * (y - np.mean(y))) / np.sum(np.power(x - np.mean(x), 2))
        a0 = np.mean(y) - a1 * np.mean(x)

        return a0, a1

    def plot_data_and_line(mileage, friction_index, a0, a1):
        plt.scatter(mileage, friction_index, label='Data Points')
        y_line = a0 + a1 * np.array(mileage)
        plt.plot(mileage, y_line, label='Least Squares Line', color='red')
        plt.xlabel('Mileage (Thousands)')
        plt.ylabel('Friction Index')
        plt.title('Mileage vs. Friction Index (Least Squares Line)')
        plt.grid(True)
        plt.legend()
        plt.show()

    mileage = [2000, 6000, 20000, 30000, 40000]
    friction_index = [20, 18, 10, 6, 2]

    a0, a1 = least_squares_line(mileage, friction_index)

    print(f"Least Squares Line Equation: y = {a0:.4f} + {a1:.4f}x")
    plot_data_and_line(mileage, friction_index, a0, a1)

    # second version
    # def linear_regression(mileage, friction_index):
    #     # Construct the X matrix
    #     X = np.column_stack((np.ones(len(mileage)), mileage))
        
    #     # Calculate coefficients using least squares
    #     coeffs = np.linalg.lstsq(X, friction_index, rcond=None)[0]
        
    #     # Calculate the predicted friction index
    #     predicted_friction_index = np.dot(X, coeffs)
        
    #     # Plot the data points and the line of best fit
    #     plt.scatter(mileage, friction_index, label='Data Points')
    #     plt.plot(mileage, predicted_friction_index, color='red', label='Line of Best Fit')
        
    #     # Add labels and legend
    #     plt.xlabel('Mileage')
    #     plt.ylabel('Friction Index')
    #     plt.title('Linear Regression: Mileage vs Friction Index')
    #     plt.legend()
        
    #     # Show plot
    #     plt.grid(True)
    #     plt.show()
        
    #     # Return coefficients
    #     return coeffs

    # # Given data
    # mileage = np.array([2000, 6000, 20000, 30000, 40000])
    # friction_index = np.array([20, 18, 10, 6, 2])

    # # Perform linear regression and plot
    # coefficients = linear_regression(mileage, friction_index)
    # print("Coefficients (a, b):", coefficients)

def exercise5():
    print()
    print()
    print("Exercise 5:")
    