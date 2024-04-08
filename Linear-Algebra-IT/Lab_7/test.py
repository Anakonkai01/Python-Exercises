import sympy as sp

def exercise2():
    print()
    print()
    print("Exercise 2:")
    vectors_a = [(1, -2, 0), (0, -4, 1), (1, -1, 1)]
    vectors_b = [(1, 0, 2), (0, 1, 4), (2, -2, -4)]
    vectors_c = [(1, -2, 3, 4), (2, 4, 5, 0), (-2, 0, 0, 4), (3, 2, 1, -1)]
    vectors_d = [(0, 0, 1, 2, 3), (0, 0, 2, 3, 1), (1, 2, 3, 4, 5), (2, 1, 0, 0, 0), (-1, -3, -5, 0, 0)]

    def check_linear_independence(vectors):
        matrix = sp.Matrix(vectors)
        rank_original = matrix.rank()
        return rank_original == len(vectors)

    def find_linear_combination_coefficients(vectors):
        matrix = sp.Matrix(vectors)
        rref_matrix, _ = matrix.rref()
        return rref_matrix

    results = []
    for vectors in [vectors_a, vectors_b, vectors_c, vectors_d]:
        is_independent = check_linear_independence(vectors)
        if is_independent:
            results.append("Linearly independent: true")
        else:
            coefficients = find_linear_combination_coefficients(vectors)
            results.append("Linearly independent: false, RREF: \n" + str(coefficients))

    print("Results:")
    for i, result in enumerate(results):
        print("{}. {}".format(chr(97 + i), result))

exercise2()
