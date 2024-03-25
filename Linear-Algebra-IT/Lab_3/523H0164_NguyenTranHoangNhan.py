import numpy as np


def exercise8():
    print()
    print()
    print("Exercise 8:")
    A = np.array([[12,15,10,16,12],[5,9,14,7,10],[8,12,10,9,15]])
    A = A.T
    B = np.array([2,1,3])
    C = np.matmul(A,B)

    print("Monday :$",C[0])
    print("Tuesday :$",C[1])
    print("Wednesday :$",C[2])
    print("Thursday :$",C[3])
    print("Friday :$",C[4])


def exercise9():
    print()
    print()
    print("Exercise 9:")
    T = np.array([[0.6,0.7],[0.4,0.3]])
    p = np.array([[0.5],[0.5]])
    Tk = T.copy()
    pk = p.copy()
    for i in range(100001):
        pk = np.matmul(Tk,p)
        Tk = np.matmul(Tk,T)
        if(i == 1 or i == 2 or i == 100 or i == 10 or i == 100000):
            print()
            print(f"p{i}:")
            print(pk)


def exercise10():
    print()
    print()
    print("Exercise 10:")
    A = np.array([[-1,4,8],[-9,1,2]])
    B = np.array([[5,8],[0,-6],[5,6]])
    C = np.array([[-4,1],[6,5]])
    D = np.array([[-6,3,1],[8,9,-2],[6,-1,5]])

    print("a:")
    print("Can not calculate")

    print()
    print("b:")
    print("Can not calculate")

    print()
    print("c:")
    print(C-C.T)

    print()
    print("d:")
    print(D-D.T)

    print()
    print("e:")
    print((D.T).T)

    print()
    print("f:")
    print(2*C.T)

    print()
    print("g:")
    print(A.T + B)

    print()
    print("h:")
    print((A.T + B).T)

    print()
    print("i:")
    print((2*A.T - 5*B).T)

    print()
    print("j:")
    print((-D).T)

    print()
    print("k:")
    print(-(D).T)

    print()
    print("l:")
    print((np.matmul(C,C)).T)

    print()
    print("m:")
    C = C.T
    print(np.matmul(C,C))

def exercise11():
    print()
    print()
    print("Exercise 11:")
    A = np.array([[2,4,1],[6,7,2],[3,5,9]])
    
    if (A.shape[0] == A.shape[1]):
        print("a: A matrix is square")
    else :
        print("a: A matrix is not square")

    if (np.allclose(A,A.T)):
        print("b: A matrix is symetric")
    else :
        print("b: A matrix is not symetric")

    if (np.allclose(-A,A.T)):
        print("c: A matrix is skew-symetric")
    else :
        print("c: A matrix is not skew-symetric")

    print("d: Upper triangular matrix:")
    print(np.triu(A,k=0))

    print("e: Lower triangular matrix:")
    print(np.tril(A,k=0))


def exercise12():
    print()
    print()
    print("Exercises 12:")
    A = np.array([6,0,0,5,1,7,2,-5,2,0,0,0,8,3,1,8])
    A = np.reshape(A,(4,4))
    B = np.array([1,-2,5,2,0,0,3,0,2,-6,-7,5,5,0,4,4])
    B = np.reshape(B,(4,4))
    C = np.array([3,5,-8,4,0,-2,3,-7,0,0,1,5,0,0,0,2])
    C = np.reshape(C,(4,4))
    D = np.array([4,0,0,0,7,-1,0,0,2,6,3,0,5,-8,3,0,5,-8,4,-3])
    D = np.reshape(D,(5,4))
    E = np.array([4,0,-7,3,-5,0,0,2,0,0,7,3,-6,4,-8,5,0,5,2,-3,0,0,9,-1,2])
    E = np.reshape(E,(5,5))
    F = np.array([6,3,2,4,0,9,0,-4,1,0,8,-5,6,7,1,3,0,0,0,0,4,2,3,2,0])
    F = np.reshape(F,(5,5))

    print("det(A) =",round(np.linalg.det(A),2))
    print("det(B) =",round(np.linalg.det(B),2))
    print("det(C) =",round(np.linalg.det(C),2))
    print("det(D) = Can not calculate")
    print("det(E) =",round(np.linalg.det(E),2))
    print("det(F) =",round(np.linalg.det(F),2))


def exercise13():
    print()
    print()
    print("Exercises 13:")
    print("No, there is no such law det(A+B)=det(A)+det(B)")
    print()
    def calculate_difference(n):
        A = np.random.randint(0, 10, size=(n, n))  
        B = np.random.randint(0, 10, size=(n, n))  

        det_A_plus_B = np.linalg.det(A + B)
        det_A = np.linalg.det(A)
        det_B = np.linalg.det(B)

        difference = round(det_A_plus_B - det_A - det_B, 2)

        return difference

    for size in range(2, 6):
        differences = [calculate_difference(size) for _ in range(3)]
        print(f"For size {size}x{size}:")
        print("Differences:", differences)
        print()


def exercise14():
    print()
    print()
    print("Exercises 14:")
    print("Yes it true det(AB)=(detA)(detB)")
    print()
    def calculate_det_product(n):
        A = np.random.randint(0, 10, size=(n, n))  
        B = np.random.randint(0, 10, size=(n, n))  
        det_AB = np.linalg.det(np.dot(A, B))
        det_A_det_B = np.linalg.det(A) * np.linalg.det(B)

        return round(det_AB, 2), round(det_A_det_B, 2)

    for _ in range(4):
        n = np.random.randint(2, 6)  
        det_AB, det_A_det_B = calculate_det_product(n)
        print(f"For {n}x{n} matrices:")
        print("det(AB):", det_AB)
        print("det(A) * det(B):", det_A_det_B)
        print()

def exercise15():
    print()
    print()
    print("Exercises 15:")
    A = np.array([2,4,5/2,-3/4,2,1/4,1/4,1/2,2])
    B = np.array([1,-1/2,3/4,3/2,1/2,-2,1/4,1,1/2])
    A = np.reshape(A,(3,3))
    B = np.reshape(B,(3,3))
    print("A^-1 * B^-1 =\n",np.matmul(np.linalg.inv(A),np.linalg.inv(B)))
    print("\n(AB)^-1 \n ",np.linalg.inv(np.matmul(A,B)))
    print("\n(BA)^-1 \n ",np.linalg.inv(np.matmul(B,A)))
    print("\n(A^-1)^T =\n ",np.linalg.inv(A))
    print("\n(A^T)^-1 =\n ",np.linalg.inv(A.T))


exercise8()
exercise9()
exercise10()
exercise11()
exercise12()
exercise13()
exercise14()
exercise15()