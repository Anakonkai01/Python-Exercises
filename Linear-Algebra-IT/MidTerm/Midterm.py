import sympy as sp
import numpy as np
import matplotlib.pyplot as plt


rows = 10
cols = 10

# Create a 1D array with sequential numbers
a_matrix = np.arange(1, rows * cols + 1)
# Reshape the 1D array into the desired shape
a_matrix = a_matrix.reshape((rows, cols))

rows = 2
cols = 10

b_matrix = np.arange(1, rows * cols + 1)
b_matrix = b_matrix.reshape((rows, cols))

rows = 10
cols = 2

c_matrix = np.arange(1, rows * cols + 1)
c_matrix = c_matrix.reshape((rows, cols))




def a():
    result = a_matrix + a_matrix.T + np.matmul(c_matrix,b_matrix) + np.matmul(b_matrix.T,c_matrix.T)
    print(result)

def b():
    result = 0
    for i in range(10):
        result += np.linalg.matrix_power(a_matrix,i+1)/(10+i)
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
    def is_prime(n):
        if n <= 1:
            return False
        for i in range(2, int(np.sqrt(n)) + 1):
            if n % i == 0:
                return False
        return True

    prime_numbers = np.array([num for row in a_matrix for num in row if is_prime(num)])
    print(prime_numbers)

def f():
    d_matrix = np.matmul(c_matrix,b_matrix)

    # Reverse elements in the odd rows of the matrix D
    for i in range(len(d_matrix)):
        if i % 2 == 0:
            d_matrix[i] = np.flip(d_matrix[i])

    # Print the resultant matrix
    print("Resultant Matrix:")
    print(d_matrix)


def g():
    def is_prime(n):
        if n <= 1:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True



    # Find the maximum number of prime numbers in any row
    max_prime_count = max(sum(is_prime(num) for num in row) for row in a_matrix)

    # Find all rows with the maximum number of prime numbers
    rows_with_max_primes = [row for row in a_matrix if sum(is_prime(num) for num in row) == max_prime_count]

    # Print the rows with the maximum number of prime numbers
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

    # Find the length of the longest contiguous odd numbers sequence in each row
    max_odd_sequence_lengths = [longest_odd_sequence(row) for row in matrix]

    # Find the maximum length of the contiguous odd numbers sequence among all rows
    max_length = max(max_odd_sequence_lengths)

    # Find the rows with the maximum length of the contiguous odd numbers sequence
    rows_with_max_length = [row for row, length in zip(matrix, max_odd_sequence_lengths) if length == max_length]

    # Print the rows with the longest contiguous odd numbers sequence
    print("Rows with the longest contiguous odd numbers sequence:")
    for row in rows_with_max_length:
        print(row)






# a()
# b()
# c()
# d()
# e()