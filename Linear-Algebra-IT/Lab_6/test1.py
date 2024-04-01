import numpy as np

def exercise1():
    a1_matrix = np.array([1,-7,-2,-3]).reshape(2,2)

    def cal_1_norm(matrix):
        col_sums = np.sum(np.abs(matrix), axis=0)
        return np.max(col_sums)

    print(cal_1_norm(a1_matrix))

def exercise2():
    b1_matrix = np.array([1,-7,-2,-3]).reshape(2,2)
    print("B1: ",np.linalg.norm(b1_matrix, ord=np.inf))

def exercise3():
    c1_matrix = np.array([5,-4,2,-1,2,3,-2,1,0]).reshape(3,3)
    print("Frobenius norm of matrix:")
    print("C1: ", np.linalg.norm(c1_matrix, 'fro'))

exercise1()
exercise2()
exercise3()
