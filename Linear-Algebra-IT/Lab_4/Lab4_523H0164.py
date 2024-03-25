import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

def exercise1():
    print("Exercise 1:")
    x,y,z,t = sp.symbols('x,y,z,t')
    eq1a = sp.Eq(x + 2*y + z,0)
    eq2a = sp.Eq(2*x - y + z, 0)
    eq3a = sp.Eq(2*x + y, 0)

    print("1a: ",sp.solve((eq1a,eq2a,eq3a),(x,y,z)))

    eq1b = sp.Eq(2*x + y + z + t,1)
    eq2b = sp.Eq(x + 2*y + z + t,1)
    eq3b = sp.Eq(x + y + 2*z + 2*t,1)
    eq4b = sp.Eq(x + y + z + 2*t,1)

    print("1b: ",sp.solve((eq1b,eq2b,eq3b,eq4b),(x,y,z,t)))


def exercise2():
    print()
    print()
    print("Exercise 2:")
    
    def plot_linear_equations(a1, b1, c1, a2, b2, c2):
        if (a1 == 0 and b1 == 0) or (a2 == 0 and b2 == 0):
            print("Can not plot")
            return
            
        x_values = np.linspace(-10, 10, 400)
        if b1 == 0:
            num = len(x_values)
            x_values_new = np.full(num, c1/a1)
            y_values1 = np.linspace(-10, 10, num)
            plt.plot(x_values_new, y_values1, label='{}x + {}y = {}'.format(a1, b1, c1))
        else:
            y_values1 = (c1 - a1 * x_values) / b1
            plt.plot(x_values, y_values1, label='{}x + {}y = {}'.format(a1, b1, c1))

        if b2 == 0:
            num = len(x_values)
            x_values_new = np.full(num, c2/a2)
            y_values2 = np.linspace(-10, 10, num)
            plt.plot(x_values_new, y_values2, label='{}x + {}y = {}'.format(a2, b2, c2))
        else:
            y_values2 = (c2 - a2 * x_values) / b2
            plt.plot(x_values, y_values2, label='{}x + {}y = {}'.format(a2, b2, c2))
        
        determinant = a1 * b2 - a2 * b1
        if determinant != 0:
            x_solution = (c1 * b2 - c2 * b1) / determinant
            y_solution = (a1 * c2 - a2 * c1) / determinant
            print("Solution: x = {}, y = {}".format(x_solution, y_solution))
            plt.scatter(x_solution, y_solution, color='red', label='Solution')
        else :
            if (a1 / a2 != b1 / b2) or (a1 / a2 != c1 / c2):
                print("No solution")
            else:
                print("Infinite solution")
        
                

        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Graph of Linear Equations')
        plt.legend()
            
        plt.show()
    
    
    #system1 
    print("System1:")
    a1,b1,c1 = 1,1,0
    a2,b2,c2 = 1,-1,2
    plot_linear_equations(a1,b1,c1,a2,b2,c2)
    
    #system2
    print("System2:")
    a1,b1,c1 = 1,1,5
    a2,b2,c2 = 2,2,12
    plot_linear_equations(a1,b1,c1,a2,b2,c2)

    #system3
    print("System3:")
    a1,b1,c1 = 1,1,5
    a2,b2,c2 = 3,3,15
    plot_linear_equations(a1,b1,c1,a2,b2,c2)

    #system4
    print("System4:")
    a1,b1,c1 = 1,0,0
    a2,b2,c2 = 1,-1,2
    plot_linear_equations(a1,b1,c1,a2,b2,c2)



def exercise3():
    print()
    print()
    print("Exercise 3:")
    def plot3D_planes(a1, b1, c1, d1, a2, b2, c2, d2, a3, b3, c3, d3):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        if c1 != 0:
            z_func1 = lambda x, y: (d1 - a1*x - b1*y) / c1
            
            x_value1 = np.linspace(-10, 10, 100)
            y_value1 = np.copy(x_value1)
            
            X1, Y1 = np.meshgrid(x_value1, y_value1)
            Z1 = z_func1(X1, Y1)
            
            ax.plot_surface(X1, Y1, Z1, alpha=0.5, rstride=100, cstride=100)
            
        else:
            if b1 != 0:
                y_func1 = lambda x: (d1 - a1*x) / b1
                x_value1 = np.linspace(-10, 10, 100)
                y_value1 = list(map(y_func1, x_value1))
                ax.plot(x_value1, y_value1, zdir='z', zs=-50, color='red')
            elif a1 != 0:
                x_value1 = np.linspace(-10, 10, 100)
                y_value1 = np.full_like(x_value1, d1/a1)
                ax.plot(x_value1, y_value1, zdir='z', zs=-50, color='green')
            else:
                print("Cannot plot the graph")
                return

        if c2 != 0:
            z_func2 = lambda x, y: (d2 - a2*x - b2*y) / c2
            
            x_value2 = np.linspace(-10, 10, 100)
            y_value2 = np.copy(x_value2)
            
            X2, Y2 = np.meshgrid(x_value2, y_value2)
            Z2 = z_func2(X2, Y2)
            
            ax.plot_surface(X2, Y2, Z2, alpha=0.5, rstride=100, cstride=100)
            
        else:
            if b2 != 0:
                y_func2 = lambda x: (d2 - a2*x) / b2
                x_value2 = np.linspace(-10, 10, 100)
                y_value2 = list(map(y_func2, x_value2))
                ax.plot(x_value2, y_value2, zdir='z', zs=-50, color='blue')
            elif a2 != 0:
                x_value2 = np.linspace(-10, 10, 100)
                y_value2 = np.full_like(x_value2, d2/a2)
                ax.plot(x_value2, y_value2, zdir='z', zs=-50, color='purple')
            else:
                print("Cannot plot the graph")
                return

        if c3 != 0:
            z_func3 = lambda x, y: (d3 - a3*x - b3*y) / c3
            
            x_value3 = np.linspace(-10, 10, 100)
            y_value3 = np.copy(x_value3)
            
            X3, Y3 = np.meshgrid(x_value3, y_value3)
            Z3 = z_func3(X3, Y3)
            
            ax.plot_surface(X3, Y3, Z3, alpha=0.5, rstride=100, cstride=100)
            
        else:
            if b3 != 0:
                y_func3 = lambda x: (d3 - a3*x) / b3
                x_value3 = np.linspace(-10, 10, 100)
                y_value3 = list(map(y_func3, x_value3))
                ax.plot(x_value3, y_value3, zdir='z', zs=-50, color='orange')
            elif a3 != 0:
                x_value3 = np.linspace(-10, 10, 100)
                y_value3 = np.full_like(x_value3, d3/a3)
                ax.plot(x_value3, y_value3, zdir='z', zs=-50, color='yellow')
            else:
                print("Cannot plot the graph")
                return

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Determine the number of solutions
        coefficients_matrix = np.array([[a1, b1, c1], [a2, b2, c2], [a3, b3, c3]])
        constants_vector = np.array([d1, d2, d3])
        augmented_matrix_rank = np.linalg.matrix_rank(np.column_stack((coefficients_matrix, constants_vector)))
        coefficient_matrix_rank = np.linalg.matrix_rank(coefficients_matrix)

        if coefficient_matrix_rank != augmented_matrix_rank:
            print("Number of solutions: No solutions")
        elif coefficient_matrix_rank == augmented_matrix_rank == 3:
            print("Number of solutions: Unique solution")
        elif coefficient_matrix_rank == augmented_matrix_rank < 3:
            print("Number of solutions: Infinitely many solutions")
        plt.show()

    #System1
   
    print("System1:")
    a1,b1,c1,d1 = 25,5,1,106.8
    a2,b2,c2,d2 = 64,8,1,177.2
    a3,b3,c3,d3 = 144,12,1,279.2
    plot3D_planes(a1,b1,c1,d1,a2,b2,c2,d2,a3,b3,c3,d3)

    print("System2:")
    a1,b1,c1,d1 = 1,1,1,10
    a2,b2,c2,d2 = 1,1,1,20
    a3,b3,c3,d3 = 3,3,3,40
    plot3D_planes(a1,b1,c1,d1,a2,b2,c2,d2,a3,b3,c3,d3)

    print("System3:")
    a1,b1,c1,d1 = 1,2,1,0
    a2,b2,c2,d2 = 2,-1,1,0
    a3,b3,c3,d3 = 2,1,0,0
    plot3D_planes(a1,b1,c1,d1,a2,b2,c2,d2,a3,b3,c3,d3)


exercise1()
exercise2()
exercise3()