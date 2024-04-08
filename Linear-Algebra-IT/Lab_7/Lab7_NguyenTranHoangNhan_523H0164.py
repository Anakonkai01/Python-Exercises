import sympy as sp
import numpy as np

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
        rank = np.linalg.matrix_rank(matrix)
        q, r = np.linalg.qr(matrix)
        basic_columns = matrix[:, np.where(np.abs(r.diagonal()) > 1e-10)[0]]
        return basic_columns
        
    def findBasicRow(matrix):
        matrix_transposed = matrix.T
        rank_transposed = np.linalg.matrix_rank(matrix_transposed)
        q_transposed, r_transposed = np.linalg.qr(matrix_transposed)
        basic_rows_transposed = matrix_transposed[:, np.where(np.abs(r_transposed.diagonal()) > 1e-10)[0]]
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


exercise1()
exercise2()
exercise3()
exercise4()
exercise5()
exercise6()

# def exercise7():
#     print()
#     print()
#     print("Exercise 7:")
