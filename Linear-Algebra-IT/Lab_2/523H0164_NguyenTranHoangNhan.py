import sympy as sp
import numpy as np
import matplotlib.pyplot as pyplot


def exercise1():
    print("Exercise 1:")
    x = np.array([1, 2, 3, 4, 5])
    b = np.array([1, 2, 3, 4, 5, 6])
    c = np.arange(1, 31)
    d = np.arange(1, 26)

    A = np.tile(x, (5, 1)).T  
    B = np.tile(b, (6, 1))
    C = c.reshape(5, 6, order='F')
    D = d.reshape(5, 5,)

    print("Matrix A:")
    print(A)
    print("\nMatrix B:")
    print(B)
    print("\nMatrix C:")
    print(C)
    print("\nMatrix D:")
    print(D)
    
def exercise2():
    print()
    print()
    print("Exercise 2:")
    a = 1
    b = 9
    random_matrix = np.random.randint(a, b+1, size=(5, 6))
    print(random_matrix)

def exercise3():
    print()
    print()
    print("Exercise 3:")
    A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

    B = np.flip(A, axis=1)

    print("Original Matrix A:")
    print(A)

    print("\nFlipped Matrix B:")
    print(B)

def exercise4():
    print()
    print()
    print("Exercise 4:")
    A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

    B = np.flip(A, axis=0)

    print("Original Matrix A:")
    print(A)

    print("\nFlipped Matrix B:")
    print(B)

def exercise5():
    print()
    print()
    print("Exercise 5:")
    Y = np.array([
    [1, 2, 16, 31, 22],
    [2, 8, 12, 21, 23],
    [4, 9, 11, 14, 25],
    [3, 6, 10, 16, 34]
    ])

    x = Y[1:2,1:4]
    y = Y[:, 2]
    y = y.reshape(-1,1)
    A = Y[1:3, 1:4]
    B = Y[:, [0, 2, 4]]

    print("(a) Vector x:")
    print(x)
    print("\n(b) Vector y:")
    print(y)
    print("\n(c) Matrix A:")
    print(A)
    print("\n(d) Matrix B:")
    print(B)

def exercise6():
    print()
    print()
    print("Exercise 6:")
    A = np.array([
    [2, 4, 1],
    [6, 7, 2],
    [3, 5, 9]
    ])

    x1 = A[0, :]
    Y = A[1:, :]

    print("(a) Vector x1:")
    print(x1)

    print("\n(b) Matrix Y:")
    print(Y)

def exercise7():
    print()
    print()
    print("Exercise 7:")
    A = np.array([
    [2, 7, 9, 7],
    [3, 1, 5, 6],
    [8, 1, 2, 5]
    ])

    B = A[:, 1::2]  
    C = A[::2, :]  
    A_reshaped = np.reshape(np.transpose(A), (4, 3))

    print("(a) Vector B (even numbered columns):")
    print(B)
    print("\n(b) Vector C (odd numbered rows):")
    print(C)
    print("Matrix A convert to 4x3:")
    print(A_reshaped)


exercise1()
exercise2()
exercise3()
exercise4()
exercise5()
exercise6()
exercise7()
