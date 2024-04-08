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

# a_matrix = np.array([[25,77,60,50,20,35,25,8,10,49],
# [86,87,54,65,42,80,12,61,22,45],
# [7,97,77,88,92,79,61,57,86,87],
# [16,49,54,42,95,57,41,18,50,12],
# [99,75,85,81,86,36,75,11,47,69],
# [59,46,39,78,45,10,32,34,34,53],
# [72,31,61,31,42,16,32,92,63,57],
# [66,2,3,14,13,52,4,98,4,51],
# [26,30,42,60,79,100,2,80,96,76],
# [18,81,61,36,29,25,35,87,73,58]])

# b_matrix = np.array([[7,12,18,11,14,8,11,7,1,15],
# [10,20,3,17,17,14,1,11,14,7]])

# c_matrix = np.array([[19,1],
# [16,10],
# [3,7],
# [6,20],
# [5,19],
# [4,3],
# [10,18],
# [12,17],
# [14,20],
# [6,5]])

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