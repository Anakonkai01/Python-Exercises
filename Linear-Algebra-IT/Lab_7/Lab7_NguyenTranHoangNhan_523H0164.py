import sympy as sp
import numpy as np
from scipy.linalg import orth,hilbert,pascal

def exercise1():
    print("Exercise 1:")
    av1 = np.array([1,2,3,4])
    av2 = np.array([-1,0,1,3])
    av3 = np.array([0,5,-6,8])
    aw = np.array([3,-6,17,11])

    bv1 = np.array([1,1,2,2])
    bv2 = np.array([2,3,5,6])
    bv3 = np.array([2,-1,3,6])
    bw = np.array([0,5,3,0])

    cv1 = np.array([1,1,2,2])
    cv2 = np.array([2,3,5,6])
    cv3 = np.array([2,-1,3,6])
    cw = np.array([-1,6,1,-4])

    dv1 = np.array([1,2,3,4])
    dv2 = np.array([-1,0,1,3])
    dv3 = np.array([0,5,-6,8])
    dv4 = np.array([1,15,-12,8])
    dw = np.array([0,-6,17,11])

    A1 = np.column_stack((av1, av2, av3, aw))
    B1 = np.column_stack((bv1, bv2, bv3, bw))
    C1 = np.column_stack((cv1, cv2, cv3, cw))
    D1 = np.column_stack((dv1, dv2, dv3, dv4, dw))

    is_linear_combination_A = np.linalg.matrix_rank(A1) == np.linalg.matrix_rank(np.column_stack((av1, av2, av3, aw)))
    is_linear_combination_B = np.linalg.matrix_rank(B1) == np.linalg.matrix_rank(np.column_stack((bv1, bv2, bv3, bw)))
    is_linear_combination_C = np.linalg.matrix_rank(C1) == np.linalg.matrix_rank(np.column_stack((cv1, cv2, cv3, cw)))
    is_linear_combination_D = np.linalg.matrix_rank(D1) == np.linalg.matrix_rank(np.column_stack((dv1, dv2, dv3, dv4, dw)))

    print("w is a linear combination of vectors A: ", is_linear_combination_A)
    print("w is a linear combination of vectors B: ", is_linear_combination_B)
    print("w is a linear combination of vectors C: ", is_linear_combination_C)
    print("w is a linear combination of vectors D: ", is_linear_combination_D)

def exercise2():
    print()
    print()
    print("Exercise 2:")
    vectors_a = [(1, -2, 0), (0, -4, 1), (1, -1, 1)]
    vectors_b = [(1, 0, 2), (0, 1, 4), (2, -2, -4)]
    vectors_c = [(1, -2, 3, 4), (2, 4, 5, 0), (-2, 0, 0, 4), (3, 2, 1, -1)]
    vectors_d = [(0, 0, 1, 2, 3), (0, 0, 2, 3, 1), (1, 2, 3, 4, 5), (2, 1, 0, 0, 0), (-1, -3, -5, 0, 0)]

    def check_linear_independence(vectors):
        matrix = np.array(vectors)
        rank_original = np.linalg.matrix_rank(matrix)
        return rank_original == len(vectors)

    def find_linear_combination_coefficients(vectors):
        matrix = np.array(vectors)
        coefficients = np.linalg.lstsq(matrix.T, np.zeros(len(vectors)), rcond=None)[0]
        return coefficients
    
    results = []
    for vectors in [vectors_a, vectors_b, vectors_c, vectors_d]:
        is_independent = check_linear_independence(vectors)
        if is_independent:
            results.append("Linearly independent: true")
        else:
            coefficients = find_linear_combination_coefficients(vectors)
            results.append("Linearly independent: false, coefficients: " + str(coefficients))

    print("Results:")
    for i, result in enumerate(results):
        print("{}. {}".format(chr(97 + i), result))

def exercise3():
    print()
    print()
    print("Exercise 3:")
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

def exercise4():
    print()
    print()
    print("Exercise 4:")
    a2_matrix = sp.Matrix([[1, 0, 2, 3], [4, -1, 0, 2], [0, -1, -8, -10]])

    null_space_basis = a2_matrix.nullspace()
    if len(null_space_basis) >= 2:
        v1 = null_space_basis[0]
        v2 = null_space_basis[1]

        a = sp.symbols('a')
        b = sp.symbols('b')

        linear_combination = a * v1 - b * v2

        is_in_null_space = a2_matrix * linear_combination == sp.zeros(3)

        print("Basis for the null space of A2:")
        print("v1:", v1)
        print("v2:", v2)
        print("\nChecking if any linear combination of v1 and v2 is in null(A2):")
        print("Linear combination:", a, "* v1 -", b, "* v2")
        print("Is the linear combination in null(A2)?", is_in_null_space)
    else:
        print("The null space has fewer than two basis vectors.")

def exercise5():
    print()
    print()
    print("Exercise 5:")
    aw = np.array([1,1,-1,-3]).reshape(4,1)
    aA_matrix = np.array([7, 6, -4, 1, -5, -1, 0, -2, 9, -11, 7, -3, 19, -9, 7, 1]).reshape(4,4)
    bw = np.array([1,2,1,0]).reshape(4,1)
    bA_matrix = np.array([-8, 5, -2, 0, -5, 2, 1, -2,10, -8, 6, -3,3, -2, 1, 0]).reshape(4,4)

    def is_in_column_space(w, A):
        projection = np.dot(A, np.linalg.pinv(A) @ w)  
        return np.allclose(w, projection)

    def is_in_null_space(w, A):
        return np.allclose(np.dot(A, w), np.zeros(len(w)))

    aw = np.array([1,1,-1,-3]).reshape(4,1)
    aA_matrix = np.array([7, 6, -4, 1, -5, -1, 0, -2, 9, -11, 7, -3, 19, -9, 7, 1]).reshape(4,4)
    bw = np.array([1,2,1,0]).reshape(4,1)
    bA_matrix = np.array([-8, 5, -2, 0, -5, 2, 1, -2,10, -8, 6, -3,3, -2, 1, 0]).reshape(4,4)

    print("a):")
    print("w is in the column space of matrix A: ",is_in_column_space(aw, aA_matrix))
    print("w is in the null space of matrix A: ",is_in_null_space(aw, aA_matrix))

    print("\nb):")
    print("w is in the column space of matrix A: ",is_in_column_space(bw, bA_matrix))
    print("w is in the null space of matrix A: ",is_in_null_space(bw, bA_matrix))

def exercise6():
    print()
    print()
    print("Exercise 6:")
    a_matrix = np.array([5, 1, 2, 2, 0, 3, 3, 2, -1, -12, 8, 4, 4, -5, 12, 2, 1, 1, 0, -2]).reshape(4, 5)

    B_columns = [0, 1, 3]  
    B_matrix = a_matrix[:, B_columns]

    a3 = a_matrix[:, 2] 
    a5 = a_matrix[:, 4] 

    rank_B = np.linalg.matrix_rank(B_matrix)
    if rank_B == B_matrix.shape[1]:
        print("The columns of B are linearly independent.")
    else:
        print("The columns of B are linearly dependent.")

    a3_in_col_space = np.allclose(a3, B_matrix @ np.linalg.lstsq(B_matrix, a3, rcond=None)[0])
    a5_in_col_space = np.allclose(a5, B_matrix @ np.linalg.lstsq(B_matrix, a5, rcond=None)[0])

    print(f"Vector a3 is in the column space of B: {a3_in_col_space}")
    print(f"Vector a5 is in the column space of B: {a5_in_col_space}")

    
def exercise7():
    print()
    print()
    print("Exercise 7:")
    # Define the vectors
    v1 = np.array([1, 0, 2])
    v2 = np.array([0, 1, 4])
    v3 = np.array([2, -2, -4])

    A = np.column_stack((v1, v2, v3))
    reduced_A = np.linalg.matrix_rank(A)
    dimension = reduced_A

    print("Dimension of the span:", dimension)
    basis = A[:, :reduced_A]
    print("Basis for the span:")
    print(basis)

def exercise8():
    print()
    print()
    print("Exercise 8:")

    # a
    hilbert_matrix = hilbert(5)
    _, _, V = np.linalg.svd(hilbert_matrix)
    null_basis_hilbert = V.T[:, -4:]

    print("Basis for the null space of Hilbert matrix:")
    print(null_basis_hilbert)

    # b
    pascal_matrix = pascal(5)
    _, _, V = np.linalg.svd(pascal_matrix)
    null_basis_pascal = V.T[:, -4:]

    print("Basis for the null space of Pascal matrix:")
    print(null_basis_pascal)

    # c
    def generate_magic_square(n):
        magic_square = np.zeros((n, n), dtype=int)
        i, j = 0, n // 2
        num = 1

        while num <= n**2:
            magic_square[i, j] = num
            num += 1
            newi, newj = (i - 1) % n, (j + 1) % n

            if magic_square[newi, newj]:
                i += 1
            else:
                i, j = newi, newj

        return magic_square

    magic_matrix = generate_magic_square(5)
    _, _, V = np.linalg.svd(magic_matrix)
    null_basis_magic = V.T[:, -4:]

    print("\nBasis for the null space of Magic matrix:")
    print(null_basis_magic)

    
def exercise9():
    print()
    print()
    print("Exercise 9:")
    u1 = (3, 1, 1)
    u2 = (-1, 2, 1)
    u3 = (-1/2, 2, 7/2)

    def is_orthogonal(vectors):
        n = len(vectors)
        for i in range(n):
            for j in range(i + 1, n):
                dot_product = sum(vectors[i][k] * vectors[j][k] for k in range(len(vectors[i])))
                if dot_product != 0:
                    return False
        return True

    vectors = [u1, u2, u3]

    print("The set of vectors is orthogonal: ",is_orthogonal(vectors))

def exercise10():
    print()
    print("Exercise 10:")
    y = np.array([7, 6])
    u = np.array([4, 2])

    def orthogonal_projection(y, u):
        dot_y_u = np.dot(y, u)
        dot_u_u = np.dot(u, u)
        projection = (dot_y_u / dot_u_u) * u
        return projection

    print("The orthogonal projection of y onto u is: ",orthogonal_projection(y,u))


def exercise11():
    print()
    print()
    print("Exercise 11:")
    U = np.array([[1, 0],
                [0, 1],
                [0, 0]])

    def has_orthonormal_columns(matrix):
        product = np.dot(matrix.T, matrix)
        identity_matrix = np.eye(matrix.shape[1])
        return np.allclose(product, identity_matrix)

    print("The matrix U has orthonormal columns: ",has_orthonormal_columns(U))

def exercise12():
    print()
    print()
    print("Exercise 12:")
    A = np.array([-10, 13, 7, -11, 2, 1, -5, 3, -6, 3, 13, -3, 16, -16, -2, 5, 2, 1, -5, -7]).reshape(5, 4)
    orthogonal_basis = orth(A)
    print("Orthogonal basis for the column space of A:")
    print(orthogonal_basis)


exercise1()
exercise2()
exercise3()
exercise4()
exercise5()
exercise6()
exercise7()
exercise8()
exercise9()
exercise10()
exercise11()
exercise12()
