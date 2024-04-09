import numpy as np

def exercise3():
    print("\nExercise 3:")
    c_matrix = np.array([1, 0, 2, 3, 4, -1, 0, 2, 0, -1, -8, -10]).reshape(3,4)

    def findBasicColumn(matrix):
        _, r = np.linalg.qr(matrix)
        # Select columns where absolute values of diagonal elements of r are greater than a threshold
        basic_columns = matrix[:, np.where(np.abs(np.diag(r)) > 1e-10)[0]]
        return basic_columns

    def findBasicRow(matrix):
        matrix_transposed = matrix.T
        _, r_transposed = np.linalg.qr(matrix_transposed)
        # Select rows where absolute values of diagonal elements of r_transposed are greater than a threshold
        basic_rows_transposed = matrix_transposed[:, np.where(np.abs(np.diag(r_transposed)) > 1e-10)[0]]
        basic_rows = basic_rows_transposed.T
        return basic_rows

    print("Basic columns:")
    print(findBasicColumn(c_matrix))
    print("\nBasic rows:")
    print(findBasicRow(c_matrix))

# Call the function
exercise3()
