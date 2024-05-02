import numpy as np
import matplotlib.pyplot as plt

def exercise1():
    A = np.array([-1, 3.5, 14,0, 5, -26, 0, 0, 2]).reshape(3,3)
    B = np.array([-2, 0, 0, 99, 0, 0, 10, -4.5, 10]).reshape(3,3)
    C = np.array([5, 5, 0, 2, 0, 2, -3, 6,0, 0, 3, -2,0, 0, 0, 5]).reshape(4,4)
    D = np.array([3, 0, 0, 0, 6, 2, 0, 0, 0, 3, 6, 0, 2, 3, 3, -5]).reshape(4,4)
    E = np.array([3, 0, 0, 0, 0, -5, 1, 0, 0, 0, 3, 8, 0, 0, 0, 0, -7, 2, 1, 0,-4, 1, 9, -2, 3]).reshape(5,5)

    eigenvalues, eigenvectors = np.linalg.eig(A);
    print(eigenvalues)
    det_A = np.prod(eigenvalues)
    print(det_A)


def exercise2():
    def A_matrix(a):
        return np.array([-6, 28, 21,4, -15, -12,8, a, 25]).reshape(3,3)
    
    a = np.array([32,31.9,31.8,32.1,32.2])

    for i in range(len(a)):
        A = A_matrix(a[i])
        characteristic_polynomial = np.poly(A)
        eigenvalues, eigenvectors = np.linalg.eig(A);
        print("Eigenvalue of A: ",eigenvalues)
        det_A = np.prod(eigenvalues)
        print("Determinant of A base on eigenvalues:",det_A)
        print("Polynomial of A is:",characteristic_polynomial)

    def graph(polynomial):
        x = np.linspace(0,3,100)
        y = polynomial[0]*x**3 + polynomial[1]*x**2 + polynomial[2]*x + polynomial[3]

        plt.plot(x,y)
    graph(characteristic_polynomial)
    plt.show()
exercise2()