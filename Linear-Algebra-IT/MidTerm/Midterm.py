import numpy as np

# Declaration
# Matrix A
rows = 10
cols = 10
a_matrix = np.arange(1, rows * cols + 1) # Create a 1D array with sequential numbers
a_matrix = a_matrix.reshape((rows, cols)) # Reshape the 1D array into the desired shape

# Matrix B
rows = 2
cols = 10
b_matrix = np.arange(1, rows * cols + 1) # Create a 1D array with sequential numbers
b_matrix = b_matrix.reshape((rows, cols)) # Reshape the 1D array into the desired shape

# Matrix C
rows = 10
cols = 2
c_matrix = np.arange(1, rows * cols + 1) # Create a 1D array with sequential numbers
c_matrix = c_matrix.reshape((rows, cols)) # Reshape the 1D array into the desired shape




def a():
    result = a_matrix + a_matrix.T + np.matmul(c_matrix,b_matrix) + np.matmul(b_matrix.T,c_matrix.T)
    print(result)

    

def b():
    result = 0
    for i in range(3):
        result += (np.linalg.matrix_power(a_matrix,i+1)/pow(10+i,i+1))
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
    print(d_matrix)


def g():
    def is_prime(n):
        if n <= 1:
            return False
        for i in range(2, int(np.sqrt(n)) + 1):
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
    max_odd_sequence_lengths = [longest_odd_sequence(row) for row in a_matrix]

    # Find the maximum length of the contiguous odd numbers sequence among all rows
    max_length = max(max_odd_sequence_lengths)

    # Find the rows with the maximum length of the contiguous odd numbers sequence
    rows_with_max_length = [row for row, length in zip(a_matrix, max_odd_sequence_lengths) if length == max_length]

    # Print the rows with the longest contiguous odd numbers sequence
    for row in rows_with_max_length:
        print(row)



# Call the function
print("Task 1a:")
a()
print("Task 1b:")
b()
print("Task 1c:")
c()
print("Task 1d:")
d()
print("Task 1e:")
e()
print("Task 1g:")
g()
print("Task 1h:")
h()