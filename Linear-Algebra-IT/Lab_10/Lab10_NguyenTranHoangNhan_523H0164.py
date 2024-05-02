import numpy as np
import matplotlib.pyplot as plt

def exercise1():
    print("Exercise 1")
    A = np.array([-1, 3.5, 14,0, 5, -26, 0, 0, 2]).reshape(3,3)
    B = np.array([-2, 0, 0, 99, 0, 0, 10, -4.5, 10]).reshape(3,3)
    C = np.array([5, 5, 0, 2, 0, 2, -3, 6,0, 0, 3, -2,0, 0, 0, 5]).reshape(4,4)
    D = np.array([3, 0, 0, 0, 6, 2, 0, 0, 0, 3, 6, 0, 2, 3, 3, -5]).reshape(4,4)
    E = np.array([3, 0, 0, 0, 0, -5, 1, 0, 0, 0, 3, 8, 0, 0, 0, 0, -7, 2, 1, 0,-4, 1, 9, -2, 3]).reshape(5,5)

    eigenvalues, eigenvectors = np.linalg.eig(A);
    det = np.prod(eigenvalues)
    print("Eigenvalues of A:",eigenvalues)
    print("Determinant of A base on eigenvalue",det)

    print()
    eigenvalues, eigenvectors = np.linalg.eig(B);
    det = np.prod(eigenvalues)
    print("Eigenvalues of B:",eigenvalues)
    print("Determinant of B base on eigenvalue",det)
    
    print()
    eigenvalues, eigenvectors = np.linalg.eig(C);
    det = np.prod(eigenvalues)
    print("Eigenvalues of C:",eigenvalues)
    print("Determinant of C base on eigenvalue",det)
    
    print()
    eigenvalues, eigenvectors = np.linalg.eig(D);
    det = np.prod(eigenvalues)
    print("Eigenvalues of D:",eigenvalues)
    print("Determinant of D base on eigenvalue",det)

    print()
    eigenvalues, eigenvectors = np.linalg.eig(E);
    det = np.prod(eigenvalues)
    print("Eigenvalues of E:",eigenvalues)
    print("Determinant of E base on eigenvalue",det)

def exercise2():
    print()
    print()
    print("Exercise 2:")
    def A_matrix(a):
        return np.array([-6, 28, 21,4, -15, -12,8, a, 25]).reshape(3,3)
    
    a = np.array([32,31.9,31.8,32.1,32.2])

    for i in range(len(a)):
        A = A_matrix(a[i])
        characteristic_polynomial = np.poly(A)
        eigenvalues, eigenvectors = np.linalg.eig(A);
        print(f"Eigenvalue of A with value of a: {a} is: ",eigenvalues)
        det_A = np.prod(eigenvalues)
        print("Determinant of A base on eigenvalues:",round(det_A))
        print("Polynomial of A is:",characteristic_polynomial)
        print()
        x = np.linspace(0,3,100)
        y = characteristic_polynomial[0]*x**3 + characteristic_polynomial[1]*x**2 + characteristic_polynomial[2]*x + characteristic_polynomial[3]
        plt.plot(x,y)
    plt.show()

def exercise3():
    print()
    print()
    print("Exercise 3:")
    M = np.array([-3,-5,-7,-2,1,0,1,5,5]).reshape(3,3)
    eigenvalues, eigenvectors = np.linalg.eig(M)
    print("Eigenvalues of M:",eigenvalues)
    print("Eigenvector of M is:\n",eigenvectors)
    # ve nha lam them cau c, cau d

def exercise4():
    print()
    print()
    print("Exercise 4:")
    A = np.array([-2,2,-3,2,1,-6,-1,-2,0]).reshape(3,3)
    eigenvalues, eigenvectors = np.linalg.eig(A)
    print("Eigenvalues of A:",eigenvalues)
    print("Eigenvectors of A:\n",eigenvectors)

    print()
    eigenvalues, eigenvectors = np.linalg.eig(np.transpose(A))
    print("Eigenvalues of A^T:",eigenvalues)
    print("Eigenvectors of A^T:\n",eigenvectors)

    print()
    eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(A))
    print("Eigenvalues of A^-1:",eigenvalues)
    print("Eigenvectors of A^-1:\n",eigenvectors)    

def exercise5():
    print()
    print()
    print("Exercise 5:")
    A1 = np.array([4,-5,2,-3]).reshape(2,2)
    A2 = np.array([0,2,0,1]).reshape(2,2)
    A3 = np.array([2,3,1,4]).reshape(2,2)
    A4 = np.array([1,2,-2,-2,5,-2,-6,6,-3]).reshape(3,3)
    A5 = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]).reshape(4,4)
    def check_diagonalize(matrix):
        row = len(matrix)
        col = len(matrix[0])
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        row_eigenvector = len(eigenvectors)
        col_eigenvector = len(eigenvectors[0])

        if(row == row_eigenvector):
            if(col == col_eigenvector):
                return "Matrix is diagonalizable"
            else:
                return "Matrix is not diagonalizable"
        else:
            return "Matrix is not diagonalizable"
        
    print("A1:",check_diagonalize(A1))
    print("A2:",check_diagonalize(A2))
    print("A3:",check_diagonalize(A3))
    print("A4:",check_diagonalize(A4))
    print("A5:",check_diagonalize(A5))

def exercise6():
    print()
    print()
    print("Exercise 6:")
    A = np.array([1,2,-2,0,3,-2,0,0,1]).reshape(3,3)
    eigenvalues, eigenvectors = np.linalg.eig(A)

    print("Eigenvalues =",eigenvalues)
    result = np.matrix.round(np.matmul(np.matmul(np.linalg.inv(eigenvectors), A),eigenvectors),0)
    print("D = P^-1*A*P =\n",result)

def exercise7():
    print()
    print()
    print("Exercise 7:")
    A1 = np.array([1,0,0,-3]).reshape(2,2)
    A2 = np.array([-5,0,0,0]).reshape(2,2)
    A3 = np.array([6**(1/2),1,0,6**(1/2)]).reshape(2,2)
    A4 = np.array([3**(1/2),2,0,3**(1/2)]).reshape(2,2)

    U,sigma,VH = np.linalg.svd(A1)
    print("Singular Value of A1:\nU\n =",U,"\nSigma =\n",sigma,"\nVH\n =",VH)
    U,sigma,VH = np.linalg.svd(A2)
    print("\nSingular Value of A2:\nU\n =",U,"\nSigma =\n",sigma,"\nVH\n =",VH)
    U,sigma,VH = np.linalg.svd(A3)
    print("\nSingular Value of A3:\nU\n =",U,"\nSigma =\n",sigma,"\nVH\n =",VH)
    U,sigma,VH = np.linalg.svd(A4)
    print("\nSingular Value of A4:\nU\n =",U,"\nSigma =\n",sigma,"\nVH\n =",VH)

def exercise8():
    print()
    print()
    print("Exercise 8:")
    B1 = np.array([-18,13,-4,4,2,19,-4,12,-14,11,-12,8,-2,21,4,8]).reshape(4,4)
    B2 = np.array([6,-8,-4,5,-4,2,7,-5,-6,4,0,-1,-8,2,2,-1,-2,4,4,-8]).reshape(4,5)

    U,sigma,VH = np.linalg.svd(B1)
    print("Matrix U of B1:\n",U)
    print("Matrix sigma of B1:\n",np.matrix.round(sigma))
    print("Matrix VH of B1:\n",VH)
    U,sigma,VH = np.linalg.svd(B2)

    print()
    print("Matrix U of B2:\n",U)
    print("Matrix sigma of B2:\n",sigma)
    print("Matrix VH of B2:\n",VH)

exercise1()
exercise2()
exercise3()
exercise4()
exercise5()
exercise6()
exercise7()
exercise8()