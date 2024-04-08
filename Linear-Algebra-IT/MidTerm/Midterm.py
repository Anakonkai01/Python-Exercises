import numpy as np
import sympy as sp
# Declaration
# Matrix A
rows = 10
cols = 10
a_matrix = np.random.randint(1, rows * cols + 1,(rows,cols))
# Matrix B
rows = 2
cols = 10
b_matrix = np.random.randint(1, rows * cols + 1,(rows,cols))
# Matrix C
rows = 10
cols = 2
c_matrix = np.random.randint(1, rows * cols + 1,(rows,cols))



def printMatrix():
    print("Matrix A: ")
    print(a_matrix)
    print("Matrix B: ")
    print(b_matrix)
    print("Matrix C: ")
    print(c_matrix)

def a():
    result = a_matrix + a_matrix.T + np.matmul(c_matrix,b_matrix) + np.matmul(b_matrix.T,c_matrix.T)
    print(result)

def b():
    result = 0
    for i in range(10):
        result += np.linalg.matrix_power(a_matrix/(10+i),i+1)
    print(result)


def c():
    odd_row = [value for index, value in enumerate(a_matrix) if index%2 == 0]
    result_matrix = np.array(odd_row)
    print(result_matrix)

def d():
    mask = a_matrix%2 != 0
    result_matrix = a_matrix[mask]
    print(result_matrix)

def e():
    prime_numbers = np.array([num for row in a_matrix for num in row if sp.isprime(num)])
    print(prime_numbers)

def f():
    d_matrix = np.matmul(c_matrix,b_matrix)
    for i in range(len(d_matrix)):
        if i % 2 == 0:
            d_matrix[i] = np.flip(d_matrix[i])
    print(d_matrix)

def g():
    max_prime_count = max(sum(sp.isprime(num) for num in row) for row in a_matrix)
    rows_with_max_primes = [row for row in a_matrix if sum(sp.isprime(num) for num in row) == max_prime_count]
    for row in rows_with_max_primes:
        print(row)

def h():
    def longest_odd_sequence(lst):
        max_length = 0
        current_length = 0
        for num in lst:
            if num % 2 != 0:
                current_length += 1
                max_length = max(max_length, current_length)
            else:
                current_length = 0
        return max_length
    max_odd_sequence_lengths = [longest_odd_sequence(row) for row in a_matrix]
    max_length = max(max_odd_sequence_lengths)
    rows_with_max_length = [row for row, length in zip(a_matrix, max_odd_sequence_lengths) if length == max_length]
    for row in rows_with_max_length:
        print(row)



# Call the function
printMatrix()
print()
print("Task 1a:")
a()
print("\nTask 1b:")
b()
print("\nTask 1c:")
c()
print("\nTask 1d:")
d()
print("\nTask 1e:")
e()
print("\nTask 1f:")
f()
print("\nTask 1g:")
g()
print("\nTask 1h:")
h()