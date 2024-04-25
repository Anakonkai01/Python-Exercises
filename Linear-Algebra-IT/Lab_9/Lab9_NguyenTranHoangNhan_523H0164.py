import sympy as sp
import numpy as np
import matplotlib.pyplot as plt


def exercise6():
    print("Exercise 6:")
    def cubic_model(x, a, b, c, d):
        return a * x**3 + b * x**2 + c * x + d

    time_days = np.array([1, 2, 3, 4, 5, 6])
    grams = np.array([2.1, 3.5, 4.2, 3.1, 4.4, 6.8])
    params, covariance = np.polyfit(time_days, grams, 3, cov=True)
    a, b, c, d = params

    x_values = np.linspace(1, 6, 100)
    y_values = cubic_model(x_values, a, b, c, d)

    plt.scatter(time_days, grams, label='Data')
    plt.plot(x_values, y_values, color='red', label='Cubic Model')
    plt.xlabel('Time in Days')
    plt.ylabel('Grams')
    plt.title('Cubic Model Fitting')
    plt.legend()
    plt.grid(True)
    plt.show()

def exercise7():
    print()
    print()
    print("Exercise 7:")
    S_2_2 = np.array([[2, 0], [0, 2]])
    S_0_5_0_5 = np.array([[0.5, 0], [0, 0.5]])
    S_1_minus_1 = np.array([[1, 0], [0, -1]])
    S_minus_1_1 = np.array([[-1, 0], [0, 1]])

    v = np.array([1, 1])

    result_a = np.dot(S_2_2, v)
    result_b = np.dot(S_0_5_0_5, v)
    result_c = np.dot(S_1_minus_1, v)
    result_d = np.dot(S_minus_1_1, v)

    # Print the results
    print("Result (a) S2,2 with λ = 2 and µ = 2:", result_a)
    print("Result (b) S0.5,0.5 with λ = 0.5 and µ = 0.5:", result_b)
    print("Result (c) S1,-1 with λ = 1 and µ = -1:", result_c)
    print("Result (d) S-1,1 with λ = -1 and µ = 1:", result_d)

def exercise8():
    print()
    print()
    print("Exercise 8:")
    def rotation_matrix(theta):
        return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    theta_a = np.pi
    R_pi = rotation_matrix(theta_a)

    theta_b = np.pi / 3
    R_pi_over_3 = rotation_matrix(theta_b)

    print("Rotation matrix Rπ:")
    print(R_pi)
    print("\nRotation matrix Rπ/3:")
    print(R_pi_over_3)

def exercise9():
    print()
    print()
    print("Exercise 9:")
    # a
    house = np.array([[-1, -1, -1.5, -1, 0.5, 0.5, 0.725, 0.725, 1,1.5,1,1,0,0,-0.5,-0.5,-1],
              [0,1,1,2,2,2.5,2.5,2,2,1,1,0,0,0.8,0.8,0,0]])

    tranMatrix = np.array([[2],[4]])
    transHouse = np.add(house,tranMatrix)
    plt.plot(house[0],house[1],'ro-')
    plt.plot(transHouse[0],transHouse[1],'bo-')
    plt.title("Translation with tx = 2 and ty = 4")
    plt.grid(True)
    plt.show()

    # b
    theta = np.pi/3
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    transHouse = np.matmul(rotation_matrix,house)
    plt.plot(house[0],house[1],'ro-')
    plt.plot(transHouse[0],transHouse[1],'bo-')
    plt.title("Rotation with pi/3")
    plt.grid(True)
    plt.show()

    # c
    scale_matrix = np.array([[2,0],[0,3]])
    transHouse = np.matmul(scale_matrix,house)
    plt.plot(house[0],house[1],'ro-')
    plt.plot(transHouse[0],transHouse[1],'bo-')
    plt.title("Scaling with Sx = 2 and Sy = 3")
    plt.grid(True)
    plt.show()

    # d
    xshearMatrix = np.array([[1,0.5],[0,1]])
    yshearMatrix = np.array([[1,0],[-1.5,1]])
    xyshearMatrix = np.array([[1,0.5],[-1.5,1]])

    transHouse = np.matmul(xshearMatrix,house)
    plt.plot(house[0],house[1],'ro-')
    plt.plot(transHouse[0],transHouse[1],'bo-')
    plt.title("Shear along x with Shear_x = 0.5")
    plt.grid(True)
    plt.show()

    transHouse = np.matmul(yshearMatrix,house)
    plt.plot(house[0],house[1],'ro-')
    plt.plot(transHouse[0],transHouse[1],'bo-')
    plt.title("Shear along x with Shear_y = -1.5")
    plt.grid(True)
    plt.show()

    transHouse = np.matmul(xyshearMatrix,house)
    plt.plot(house[0],house[1],'ro-')
    plt.plot(transHouse[0],transHouse[1],'bo-')
    plt.title("Shear along x and y with Shear_x = 0.5, Shear_y = -1.5")
    plt.grid(True)
    plt.show()

def exercise10():
    print()
    print()
    print("Exercise 10:")
    triangle = np.array([[1,3,1,1],[1,1,3,1]])
    identity_matrix = np.array([[1,0],[0,1]])
    trans = np.matmul(-identity_matrix,triangle)
    plt.plot(triangle[0],triangle[1],'ro-')
    plt.plot(trans[0],trans[1],'bo-')
    plt.title("Shear along x and y with Shear_x = 0.5, Shear_y = -1.5")
    plt.grid(True)
    plt.show()

def exercise11():
    figure_F = np.array([[3,3,0,0,1,1,2.5,2.5,1,1,3],[4,5,5,0,0,1.8,1.8,2.8,2.8,4,4]])
    t1 = np.array([-1,0,0,0,-1,0,0,0,1]).reshape(3,3)
    t2 = np.array([1/4,0,0,0,1/4,0,0,0,1]).reshape(3,3)
    t3 = np.array([1,0,5/4,0,1,-5/4,0,0,1]).reshape(3,3)
    t4 = np.array([1/4,0,5/4,0,1/4,-5/4,0,0,1]).reshape(3,3)
    t5 = np.array([-1/4,0,0,0,-1/4,0,0,0,1]).reshape(3,3)
    t6 = np.array([-1/4,0,-5/4,0,-1/4,5/4,0,0,1]).reshape(3,3)
    t7 = np.array([1,-1,5/4,0,0,5/4,0,0,1]).reshape(3,3)

    tb = np.array([1,0,-2,0,-2,1,9,0,1]).reshape(3,3)

    # a
    figure_F = np.vstack((figure_F,np.ones(figure_F.shape[1])))
    trans = np.matmul(t1,figure_F)
    plt.plot(figure_F[0],figure_F[1],'ro-')
    plt.plot(trans[0],trans[1],'bo-')
    plt.title("Transformation figure F with T1")
    plt.grid(True)
    plt.show()

    trans = np.matmul(t2,figure_F)
    plt.plot(figure_F[0],figure_F[1],'ro-')
    plt.plot(trans[0],trans[1],'bo-')
    plt.title("Transformation figure F with T2")
    plt.grid(True)
    plt.show()

    trans = np.matmul(t3,figure_F)
    plt.plot(figure_F[0],figure_F[1],'ro-')
    plt.plot(trans[0],trans[1],'bo-')
    plt.title("Transformation figure F with T3")
    plt.grid(True)
    plt.show()

    trans = np.matmul(t4,figure_F)
    plt.plot(figure_F[0],figure_F[1],'ro-')
    plt.plot(trans[0],trans[1],'bo-')
    plt.title("Transformation figure F with T4")
    plt.grid(True)
    plt.show()

    trans = np.matmul(t5,figure_F)
    plt.plot(figure_F[0],figure_F[1],'ro-')
    plt.plot(trans[0],trans[1],'bo-')
    plt.title("Transformation figure F with T5")
    plt.grid(True)
    plt.show()

    trans = np.matmul(t6,figure_F)
    plt.plot(figure_F[0],figure_F[1],'ro-')
    plt.plot(trans[0],trans[1],'bo-')
    plt.title("Transformation figure F with T6")
    plt.grid(True)
    plt.show()

    trans = np.matmul(t7,figure_F)
    plt.plot(figure_F[0],figure_F[1],'ro-')
    plt.plot(trans[0],trans[1],'bo-')
    plt.title("Transformation figure F with T7")
    plt.grid(True)
    plt.show()

    # b
    rotationMatrix = np.array([[0,1],[-1,0]])
    originMatrix = np.array([[3,3,0,0,1,1,2.5,2.5,1,1,3],[4,5,5,0,0,1.8,1.8,2.8,2.8,4,4]])

    figure_F_diagram2 = np.matmul(rotationMatrix,originMatrix)
    figure_F_diagram2 = np.vstack((figure_F_diagram2,np.ones(figure_F_diagram2.shape[1])))
    originMatrix = np.vstack((originMatrix,np.ones(originMatrix.shape[1])))
    trans = np.matmul(tb,originMatrix)
    plt.plot(trans[0],trans[1],'bo-')
    plt.plot(figure_F_diagram2[0],figure_F_diagram2[1],'ro-')
    plt.title("Implement task 11b")
    plt.grid(True)
    plt.show()
exercise11()