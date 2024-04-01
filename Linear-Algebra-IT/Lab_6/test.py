import numpy as np

def exercise1():
    a1_matrix = np.array([1,-7,-2,-3]).reshape(2,2)

    def cal_1_norm(matrix):
        col_sums = np.sum(np.abs(matrix), axis=1)
        return np.max(col_sums)

    print(cal_1_norm(a1_matrix))

def exercise2():
    b1_matrix = np.array([1,-7,-2,-3]).reshape(2,2)

    

    print(cal_inf_norm(b1_matrix))

def exercise3():
    c1_matrix = np.array([5,-4,2,-1,2,3,-2,1,0]).reshape(3,3)

    def Frobenius_norm(matrix):
        sum = np.sum(matrix ** 2)
        return np.sqrt(sum)

    print("Frobenius norm of matrix:")
    print("C1: ",Frobenius_norm(c1_matrix))

exercise1()
exercise2()
exercise3()
