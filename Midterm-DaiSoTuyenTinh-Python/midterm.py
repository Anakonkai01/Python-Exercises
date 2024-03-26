import sympy as sp 
import numpy as np
import matplotlib.pyplot as plt

a_matrix = np.arange(1,101)
a_matrix = a_matrix.reshape(10,10)

def c():
    odd_row = [value for index, value in enumerate(a_matrix) if index%2 == 0]
    result_matrix = np.array(odd_row)
    print(result_matrix)

def d():
    mask = a_matrix%2 != 0
    # print(mask)

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
