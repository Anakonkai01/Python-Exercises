import numpy as np
import matplotlib.pyplot as plt

def exercise1():
    print("Exercise 1:")    
    A = np.array([1,3,2,4,1,6]).reshape(3,2)
    b = np.array([4,1,3]).reshape(3,1)

    x = np.linalg.linalg.lstsq(A,b,rcond = None)
    print(f"The least square solution to Ax = b is:\n{x[0]}")

def exercise2():
    print()
    print()
    print("Exercise 2:")

    A = np.array([0, 0, 1, 0, 1, 1, 1, 2, 1, 1, 0, 1, 4, 1, 1, 4, 2, 1]).reshape(6, 3)
    b = np.array([0.5, 1.6, 2.8, 0.8, 5.1, 5.9])

    x = np.linalg.lstsq(A, b, rcond=None)

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
        plt.scatter(x, y)
        a1 = np.sum((x - np.mean(x)) * (y - np.mean(y))) / np.sum(np.power(x - np.mean(x), 2))
        a0 = np.mean(y) - a1 * np.mean(x)

        x = np.linspace(min(y)-2,max(x)+2,100)
        y_line = a0 + a1 * np.array(x)
        plt.plot(x, y_line,label='Least Squares Line', color='red')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('The least squares line')
        plt.grid(True)
        plt.show()
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
        plt.show()

    mileage = [2000, 6000, 20000, 30000, 40000]
    friction_index = [20, 18, 10, 6, 2]

    a0, a1 = least_squares_line(mileage, friction_index)

    print(f"Least Squares Line Equation: y = {a0:.4f} + {a1:.4f}x")
    plot_data_and_line(mileage, friction_index, a0, a1)

def exercise5():
    print()
    print()
    print("Exercise 5:")
    data_points = np.array([(1, 7.9), (2, 5.4), (3, -9)])

    x_values = data_points[:, 0]
    y_values = data_points[:, 1]

    A = np.column_stack((np.cos(x_values), np.sin(x_values)))
    coeffs = np.linalg.lstsq(A, y_values, rcond=None)[0]

    print("A = :", coeffs[0])
    print("B = :", coeffs[1])

    A_opt = coeffs[0]
    B_opt = coeffs[1]

    def calculate_y(x):
        return A_opt * np.cos(x) + B_opt * np.sin(x)

    x_range = np.linspace(min(x_values)-1, max(x_values)+1, 100)
    y_range = calculate_y(x_range)

    plt.scatter(x_values, y_values, color='red')
    plt.plot(x_range, y_range)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Least Squares Fit: y = Acos(x) + Bsin(x)')
    plt.grid(True)
    plt.show()




exercise1()
exercise2()
exercise3()
exercise4()
exercise5()
