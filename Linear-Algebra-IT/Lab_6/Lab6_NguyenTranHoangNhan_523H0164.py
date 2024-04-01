import numpy as np

def exercise1():
    print()
    print()
    print("Exercise 1:")
    a1_matrix = np.array([1,-7,-2,-3]).reshape(2,2)
    a2_matrix = np.array([-2,8,3,1]).reshape(2,2)
    a3_matrix = np.array([2,-8,3,1]).reshape(2,2)
    a4_matrix = np.array([2,3,1,-1]).reshape(2,2)
    a5_matrix = np.array([5,-4,2,-1,2,3,-2,1,0]).reshape(3,3)

    def cal_1_norm(matrix):
        col_sums = np.sum(np.abs(matrix), axis=0)
        return np.max(col_sums)

    print("A1: ",cal_1_norm(a1_matrix))
    print("A1: ",cal_1_norm(a2_matrix))
    print("A1: ",cal_1_norm(a3_matrix))
    print("A1: ",cal_1_norm(a4_matrix))
    print("A1: ",cal_1_norm(a5_matrix))

def exercise2():
    print()
    print()
    print("Exercise 2:")
    b1_matrix = np.array([1,-7,-2,-3]).reshape(2,2)
    b2_matrix = np.array([3,6,1,0]).reshape(2,2)
    b3_matrix = np.array([5,-4,2,-1,2,3,-2,1,0]).reshape(3,3)
    b4_matrix = np.array([3,6,-1,3,1,0,2,4,-7]).reshape(3,3)
    b5_matrix = np.array([-3,0,0,0,4,0,0,0,2]).reshape(3,3)

    print("B1: ",np.linalg.norm(b1_matrix, ord=np.inf))
    print("B2: ",np.linalg.norm(b2_matrix, ord=np.inf))
    print("B3: ",np.linalg.norm(b3_matrix, ord=np.inf))
    print("B4: ",np.linalg.norm(b4_matrix, ord=np.inf))
    print("B5: ",np.linalg.norm(b5_matrix, ord=np.inf))

def exercise3():
    print()
    print()
    print("Exercise 3:")
    c1_matrix = np.array([5,-4,2,-1,2,3,-2,1,0]).reshape(3,3)
    c2_matrix = np.array([1,7,3,4,-2,-2,-2,-1,1]).reshape(3,3)
    c3_matrix = np.array([2,3,1,-1]).reshape(2,2)

    print("Frobenius norm of matrix:")
    print("C1: ",np.linalg.norm(c1_matrix, 'fro'))
    print("C2: ",np.linalg.norm(c2_matrix, 'fro'))
    print("C3: ",np.linalg.norm(c3_matrix, 'fro'))


def exercise4():
    print()
    print()
    print("Exercise 4:")
    ua = np.array([1,1]).reshape(2,1)
    va = np.array([0,1]).reshape(2,1)
    ub = np.array([1,0]).reshape(2,1)
    vb = np.array([0,1]).reshape(2,1)
    uc = np.array([-2,3]).reshape(2,1)
    vc = np.array([1/2,-1/2]).reshape(2,1)

    def findAngle(u,v):
        u = np.reshape(u,-1)
        v = np.reshape(v,-1)
        numerator = np.dot(u,v)
        denomerator = np.linalg.norm(u,2)*np.linalg.norm(v,2)
        return numerator/denomerator
    print("Angle of u,v in 4a: ",np.round(np.degrees(np.arccos(findAngle(ua,va))),2))
    print("Angle of u,v in 4b: ",np.round(np.degrees(np.arccos(findAngle(ub,vb))),2))
    print("Angle of u,v in 4c: ",np.round(np.degrees(np.arccos(findAngle(uc,vc))),2))

def exercise5():
    print()
    print()
    print("Exercise 5:")
    a = np.array([2,3])
    b = np.array([1,2,3])
    c = np.array([1/2,-1/2,1/4])
    d = np.array([2**1/2,2,-2**1/2,2**1/2])

    def findUnit(vector):
        denomerator = np.linalg.norm(vector,2)
        returnVector = []
        for value in vector:
            returnVector.append(value/denomerator)
        return returnVector
    

    print("Unit vector 5a: ",findUnit(a))
    print("Unit vector 5b: ",findUnit(b))
    print("Unit vector 5c: ",findUnit(c))
    print("Unit vector 5d: ",findUnit(d))

def exercise6():
    print()
    print()
    print("Exercise 6:")
    v1 = np.array([1,2,3])
    s2 = np.array([7,4,3])
    s3 = np.array([2,1,9])

    def findDistanceEuclidean(vector1,vector2):
        return np.linalg.norm(vector1 - vector2,2)
    print("Distance between v1 and s1:",findDistanceEuclidean(v1,s2))
    print("Distance between v1 and s2:",findDistanceEuclidean(v1,s3))
    print("Distance between s1 and s2:",findDistanceEuclidean(s2,s3))

def exercise7():
    print()
    print()
    print("Exercise 7:")
    E_matrix = np.array([80,98,99,85,106,94,71,92,76,95,100,92,124,163,140,160,176,161]).reshape(3,6)
    A_matrix = np.array([1,2,3,2,1,2,3,2,4]).reshape(3,3)

    decode_matrix = np.dot(np.linalg.inv(A_matrix),E_matrix)
    for row in decode_matrix.T:
        for value in row:
            if(int(round(value))+61 == 91):
                print(" ",end="")
            else:
                print(chr(int(round(value))+61),end="")
            

def exercise8():
    print()
    print()
    print("Exercise 8:")
    str1 = "ATTACK"
    str2 = "LINEAR ALGEBRA LABORATORY"
    key_matrix = np.array([3,4,5,1,3,1,1,1,2]).reshape(3,3)
    
    def encode_message (str):
        d_matrix = [ord(char) for char in str]
        
        while(1):
            if len(d_matrix)%len(key_matrix) == 0:
                colD = int(len(d_matrix)/len(key_matrix))
                d_matrix = np.array(d_matrix)
                d_matrix = np.reshape(d_matrix,(3,colD))
                break
            else:
                d_matrix.append(31)

        d_matrix = np.abs(d_matrix - 61)
        def map_index(row, col, num_rows=3):
            return col * num_rows + row

        new_d_matrix = np.zeros_like(d_matrix)

        for row in range(d_matrix.shape[0]):
            for col in range(d_matrix.shape[1]):
                new_index = map_index(row, col)
                new_d_matrix[row, col] = d_matrix.flat[new_index]
        
        return np.matmul(key_matrix,new_d_matrix)

    print("encode message 'ATTACK':")
    print(encode_message(str1))
    print("encode message 'LINEAR ALGEBRA LABORATORY':")
    print(encode_message(str2))
    

def exercise9():
    print()
    print()
    print("Exercise 9:")
    def cosine_similarity(doc_matrix):

        doc_magnitudes = np.linalg.norm(doc_matrix, axis=1)

        doc_magnitudes[doc_magnitudes == 0] = 1e-8

        similarity_matrix = np.dot(doc_matrix, doc_matrix.T) / (np.outer(doc_magnitudes, doc_magnitudes))

        print("Cosine Similarity:")

        print("       |", end="")
        for i in range(doc_matrix.shape[0]):
            print(f"Doc {i+1} |", end="")
        print()
        print("------" * (len(str(doc_matrix.shape[0])) * 5 + 2))

        for i in range(doc_matrix.shape[0]):
            print(f" Doc {i+1} |", end="")
            for j in range(doc_matrix.shape[0]):
                print(f"{similarity_matrix[i, j]:.3f} |", end="")
            print()
            if i != doc_matrix.shape[0] - 1:
                print("------" * (len(str(doc_matrix.shape[0])) * 5 + 2))

    doc_matrix = np.array([[0, 4, 0, 0, 0, 2, 1, 3],
                        [3, 1, 4, 3, 1, 2, 0, 1],
                        [3, 0, 0, 0, 3, 0, 3, 0],
                        [0, 1, 0, 3, 0, 0, 2, 0],
                        [2, 2, 2, 3, 1, 4, 0, 2]])

    cosine_similarity(doc_matrix)

def exercise10():
    print()
    print()
    print("Exercise 10:")
    def cosine_similarity(doc_matrix, query_vector):
        query_vector = query_vector.reshape(1, -1)  

        doc_magnitudes = np.linalg.norm(doc_matrix, axis=1)

        doc_magnitudes[doc_magnitudes == 0] = 1e-8
        query_magnitude = np.linalg.norm(query_vector)
        if query_magnitude == 0:
            query_magnitude = 1e-8

        similarity_scores = np.dot(query_vector, doc_matrix.T) / (query_magnitude * doc_magnitudes)

        return similarity_scores

    def find_nearest_document(doc_matrix, query_vector):
        similarity_scores = cosine_similarity(doc_matrix, query_vector)

        nearest_doc_index = np.argmax(similarity_scores, axis=1)[0]

        return nearest_doc_index

    doc_matrix = np.array([[1.0, 0.5, 0.3, 0, 0, 0],
                        [0.5, 1.0, 0, 0, 0, 0],
                        [0, 0.8, 0.7, 0, 0, 0],
                        [0, 0.9, 1.0, 0.5, 0, 0],
                        [0, 0, 0, 1.0, 0, 1.0],
                        [0, 0, 0, 0, 0.7, 0],
                        [0.5, 0, 0.7, 0, 0, 0.9]])
    query_vector = np.array([0, 0, 0.7, 0.5, 0, 0.3])

    nearest_doc_index = find_nearest_document(doc_matrix, query_vector)
    print("Nearest Document Index:", nearest_doc_index + 1)



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