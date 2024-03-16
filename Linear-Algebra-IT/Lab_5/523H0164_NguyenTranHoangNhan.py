import matplotlib.pyplot as plt
import sympy as sp
import numpy as np



def exercise6():
    print("Exercise 6:")
    a0, a1, a2 = sp.symbols('a0 a1 a2')
    data = [(1, 6), (2, 15), (3, 38)]

    equations = []
    for x_val, y_val in data:
        equation = a0 + a1*x_val + a2*x_val**2 - y_val
        equations.append(equation)

    solution = sp.solve(equations, (a0, a1, a2))

    a0_val = solution[a0]
    a1_val = solution[a1]
    a2_val = solution[a2]

    print("a0:", a0_val)
    print("a1:", a1_val)
    print("a2:", a2_val)


def exercise7():
    print()
    print()
    print("Exercise 7:")
    A = np.array([[3, 3.2],
              [3.5, 3.6]])

    b = np.array([118.4, 135.2])

    solution = np.linalg.solve(A, b)

    children = round(solution[0],)
    adults = round(solution[1],)

    print("Number of children:", children)
    print("Number of adults:", adults)



def exercise8():
    print()
    print()
    print("Exercise 8:")
    x, y, z, t = sp.symbols('x y z t')

    # Define the equations
    equation1 = 2*x - 4*y + 4*z + 0.077*t - 3.86
    equation2 = -2*y + 2*z - 0.056*t + 3.47  # Adjusted the sign to move constant term to the right-hand side
    equation3 = 2*x - 2*y

    # Solve the equations
    solution = sp.solve((equation1, equation2, equation3), (x, y, z, t))

    # Print the solution
    print("Solution:")
    print(solution)

def exercise9():
    print()
    print()
    print("Exercise 9:")
    A = np.array([[0.61, 0.29, 0.15],
                [0.35, 0.59, 0.063],
                [0.04, 0.12, 0.787]])

    # Find the inverse of matrix A
    A_inv = np.linalg.inv(A)

    # Define the CIE color vector
    CIE_color = np.array([X, Y, Z])  # Replace X, Y, Z with the CIE color values

    # Convert CIE to RGB
    RGB_color = np.dot(A_inv, CIE_color)

    # Print the RGB color values
    print("RGB Color:", RGB_color)

exercise9()
def exercise10():
    print()
    print()
    print("Exercise 10:")
    A = np.array([[0.25, 0.15, 0.1],
                [0.2, 0.3, 0.15],
                [0.1, 0.1, 0.2]])

    d = np.array([100, 100, 100]).reshape(3, 1)

    I = np.identity(3)
    p = np.linalg.inv(I - A).dot(d)

    print("Production vector p:")
    print(p)

def exercise11():
    print()
    print()
    print("Exercise 11:")

    
    x1, x2, x3, x4 = sp.symbols('x1 x2 x3 x4')

    equations = [
        3*x1 - x3,
        8*x1 - 2*x4,
        2*x2 - 2*x3 - x4
    ]

    solution = sp.solve(equations, (x1, x2, x3, x4))

    for key, value in solution.items():
        print(f"{key}: {value}")


exercise11()