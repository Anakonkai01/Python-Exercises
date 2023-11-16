import sympy as sp 
import math  
import numpy as np 
import matplotlib.pyplot as plt
from sympy import AccumBounds



# exercise 1 
def exercise1():
    x = sp.symbols('x')

    def a1():
        fx = abs(x*x - x - 7) 
        lm = sp.limit(fx, x, 3) 
        print(lm)
    def b1():
        fx = abs(x-1)/(x**2-1)
        lm = sp.limit(fx,x,1)
        print(lm)
    def c1():
        fx = math.e**1.0/x
        lm = sp.limit(fx,x,1)
        print(lm)
    def d1():
        fx = (x**4 - 16)/(x - 2)
        lm = sp.limit(fx,x,2)
        print(lm)
    def e1():
        fx = (x**3 - x**2 - 5*x  - 3 )/((x - 1)**2)
        lm = sp.limit(fx,x,-1)
        print(lm)
    def f1():
        fx = (x**2 - 9)/((x**2+7)**1/2 -4)
        lm = sp.limit(fx,x,3)
        print(lm)
    def g1():
        fx = (abs(x))/(math.sin(math.radians(x)))
        lm = sp.limit(fx,x,1)
        print(lm)
    def h1():
        fx = (1-math.cos(math.radians(x)))/(x*math.sin(math.radians(x)))
        lm = sp.limit(fx,x,0)
        print(lm)
    def i1():
        fx = (2*x**2)/(3 - 3*math.cos(math.radians(x)))
        lm = sp.limit(fx,x,0)
        print(lm)
    def j1():
        fx = ((3+x)/(-1 + x))**x
        lm = sp.limit(fx,x,sp.oo)
        print(lm)
    def k1():
        fx = (1-(2)/(3 + x))**x
        lm = sp.limit(fx,x,sp.oo)
        print(lm)
    def l1():
        fx = ((1)/(x))**(1/x)
        lm = sp.limit(fx,x,sp.oo)
        print(lm)
    def m1():
        fx = (-(x**1/3) + (1+x)**1/3 )/(-(x**1/2) + (1+x)**1/2)
        lm = sp.limit(fx,x,sp.oo)
        print(lm)
    def n1():
        fx = math.factorial(x)/(x**x)
        lm = sp.limit(fx,x,sp.oo)
        print(lm)


def exercise2():
    def a():
        def fx(x):   
            return np.abs(x**2 - x - 7)

        x_values = np.arange(2,4,0.01)  # list 
        y_values = fx(x_values) #list
        plt.plot(x_values, y_values, label='|x^2 - x - 7|')
        limit_points_x = 3
        limit_points_y = fx(3)
        plt.plot(limit_points_x, limit_points_y, 'ro')
        plt.title('Graph of |x^2 - x - 7| and Limit Points')
        plt.xlabel('x')
        plt.ylabel('|x^2 - x - 7|')
        plt.legend()
        plt.grid(color='gray', linestyle='--')
        plt.show()

    def b():
        def fx(x):   
            return np.abs(x-1)/(x**2-1)

        x_values = np.arange(2,4,0.001)   
        y_values = fx(x_values) 
        plt.plot(x_values, y_values, label='abs(x-1)/(x**2-1)|x^2 - x - 7|')
        limit_points_x = 1
        limit_points_y = fx(1) 
        plt.plot(limit_points_x, limit_points_y, 'ro')
        plt.title('Graph of abs(x-1)/(x**2-1) and Limit Points')
        plt.xlabel('x')
        plt.ylabel('abs(x-1)/(x**2-1)')
        plt.legend()
        plt.grid(color='gray', linestyle='--')
        plt.show()

    def c():
        def fx(x):   
            return math.e**1.0/x
        x_values = np.arange(2,4,0.001)
        y_values = fx(x_values)
        plt.plot(x_values, y_values, label='e^1.0/x')
        limit_points_x = 1
        limit_points_y = fx(1)
        plt.plot(limit_points_x, limit_points_y, 'ro')
        plt.title('Graph of e^1.0/x and Limit Points')
        plt.xlabel('x')
        plt.ylabel('e^1.0/x')
        plt.legend()
        plt.grid(color='gray', linestyle='--')
        plt.show()

    def d():
        def fx(x):
            return (x**4 - 16)/(x - 2)
        x_values = np.arange(2,4,0.001)
        y_values = fx(x_values)
        plt.plot(x_values, y_values, label='(x**4 - 16)/(x - 2)')
        limit_points_x = 2
        limit_points_y = fx(2)
        plt.plot(x_values,y_values, label='(x**4 - 16)/(x - 2)')
        plt.plot(limit_points_x, limit_points_y, 'ro')
        plt.title('Graph of (x**4 - 16)/(x - 2) and Limit Points')
        plt.xlabel('x')
        plt.ylabel('(x**4 - 16)/(x - 2)')
        plt.legend()
        plt.grid(color='gray', linestyle='--')
        plt.show()

    def e():
        def f(x):
            return (x**3 - x**2 - 5*x  - 3 )/((x - 1)**2)
        x_values = np.arange(2,4,0.001)
        y_values = f(x_values)
        plt.plot(x_values, y_values, label='(x**3 - x**2 - 5*x  - 3 )/((x - 1)**2)')
        limit_points_x = -1
        limit_points_y = f(limit_points_x)
        plt.plot(limit_points_x, limit_points_y, 'ro')
        plt.title('Graph of (x**3 - x**2 - 5*x  - 3 )/((x - 1)**2) and Limit Points')
        plt.xlabel('x')
        plt.ylabel('(x**3 - x**2 - 5*x  - 3 )/((x - 1)**2)')
        plt.legend()
        plt.grid(color='gray', linestyle='--')
        plt.show()

    def f():
        def f(x):
            return (x**2 - 9)/((x**2+7)**1/2 -4)
        x_values = np.arange(2,4,0.001)
        y_values = f(x_values)
        plt.plot(x_values, y_values, label='(x**2 - 9)/((x**2+7)**1/2 -4)')
        limit_points_x = 3
        limit_points_y = f(limit_points_x)
        plt.plot(limit_points_x, limit_points_y, 'ro')
        plt.title('Graph of (x**2 - 9)/((x**2+7)**1/2 -4) and Limit Points')
        plt.xlabel('x')
        plt.ylabel('(x**2 - 9)/((x**2+7)**1/2 -4)')
        plt.legend()
        plt.grid(color='gray', linestyle='--')
        plt.show()

    def g():
        def f(x):
            return (abs(x))/(math.sin(math.radians(x)))
        x_values = np.arange(2,4,0.001)
        y_values = f(x_values)
        plt.plot(x_values, y_values, label='abs(x)/sin(radians(x))')
        limit_points_x = 1
        limit_points_y = f(limit_points_x)
        plt.plot(limit_points_x, limit_points_y, 'ro')
        plt.title('Graph of abs(x)/sin(radians(x)) and Limit Points')
        plt.xlabel('x')
        plt.ylabel('abs(x)/sin(radians(x))')
        plt.legend()
        plt.grid(color='gray', linestyle='--')
        plt.show()
    def h():
        def f(x):
            return (1-math.cos(math.radians(x)))/(x*math.sin(math.radians(x)))
        x_values = np.arange(2,4,0.001)
        y_values = f(x_values)
        plt.plot(x_values, y_values, label='(1-cos(radians(x)))/(x*sin(radians(x)))')
        limit_points_x = 0
        limit_points_y = f(limit_points_x)
        plt.plot(limit_points_x, limit_points_y, 'ro')
        plt.title('Graph of (1-cos(radians(x)))/(x*sin(radians(x))) and Limit Points')
        plt.xlabel('x')
        plt.ylabel('(1-cos(radians(x)))/(x*sin(radians(x)))')
        plt.legend()
        plt.grid(color='gray', linestyle='--')
        plt.show()









def exercise3():
    def a():
        x = sp.symbols('x')
        f = 1 / (1 + 2**(1/x))
        lmRight = sp.limit(f, x, 0, '+')
        lmLeft = sp.limit(f, x, 0, '-')
        lm = sp.limit(f, x, 0)

        if isinstance(lmRight, AccumBounds):
            print("ERROR: there is no limit from the right side")
        else:
            print("Limit from the right to the point =", lmRight)

        if isinstance(lmLeft, AccumBounds):
            print("ERROR: there is no limit from the left side")
        else:
            print("Limit from the left to the point =", lmLeft)
        
        if isinstance(lm, AccumBounds):
            print("Error: unable to calculate the limit at the point")
        else:
            print("Limit at the point =", lm)
        
        if lmRight == lmLeft and lmRight == lm and lmLeft == lm:
            def f(x):
                return 1 / (1 + 2**(1/x))
            
            print("The limit of the function exists ")
            x_values = np.linspace(0.01, 10, 500)  # Avoid x=0 for this function
            y_values = f(x_values)
            plt.plot(x_values, y_values, label='1 / (1 + 2^(1/x))')
            plt.xlabel('x-axis')
            plt.ylabel('y-axis')
            plt.title('Plot of the Function')
            plt.legend()
            plt.grid()
            plt.show()
        else:
            print("Function does not exist")


    def b():
        x = sp.symbols('x')
        f3_2 = (x**2 + x) * ((x**3 + x**2)**(1/2))
        lmRight = sp.limit(f3_2, x, 0, '+')
        lmLeft = sp.limit(f3_2, x, 0, '-')
        lm = sp.limit(f3_2, x, 0)

        if isinstance(lmRight, AccumBounds):
            print("ERROR: there is no limit from the right side")
        else:
            print("Limit from the right to the point =", lmRight)

        if isinstance(lmLeft, AccumBounds):
            print("ERROR: there is no limit from the left side")
        else:
            print("Limit from the left to the point =", lmLeft)
        
        if isinstance(lm, AccumBounds):
            print("Error: unable to calculate the limit at the point")
        else:
            print("Limit at the point =", lm)
        
        if lmRight == lmLeft and lmRight == lm and lmLeft == lm:
            def f3_2(x):
                return (x**2 + x) * ((x**3 + x**2)**(1/2))
            
            print("The limit of the function exists ")
            x_values = np.arange(0, 10, 0.01)
            y_values = f3_2(x_values)
            plt.plot(x_values, y_values, label='(x^2 + x) * sqrt(x^3 + x^2)')
            plt.xlabel('x-axis')
            plt.ylabel('y-axis')
            plt.title('Plot of the Function')
            plt.legend()
            plt.grid()
            plt.show()
        else:
            print("Function does not exist")







def exercise5():
    def a():
        x = sp.symbols('x') # Định nghĩa biến ký hiệu và hàm số
        f = x**2 - 7
        c = 1 # Điểm cần kiểm tra tính liên tục
        f_c = f.subs(x, c) # Tính giá trị của hàm tại điểm c
        # Tính giới hạn của hàm khi x tiến đến c từ cả hai hướng
        limit_from_left = sp.limit(f, x, c, '-')
        limit_from_right = sp.limit(f, x, c, '+')
        # Tính giới hạn của hàm khi x tiến đến c 
        limit_fx = sp.limit(f, x, c)
        # Kiểm tra tính liên tục
        #if f_c == limit_from_left == limit_from_right:
        if f_c == limit_fx:
            print("The function is continuous at x =",c, ".")
        else:
            print(f"The function is not continuous at x = {c}.")
    def b():
        x = sp.symbols('x') # Định nghĩa biến ký hiệu và hàm số
        f = (x*2 - 3)**1/2
        c = 2 # Điểm cần kiểm tra tính liên tục
        f_c = f.subs(x, c) # Tính giá trị của hàm tại điểm c
        # Tính giới hạn của hàm khi x tiến đến c từ cả hai hướng
        limit_from_left = sp.limit(f, x, c, '-')
        limit_from_right = sp.limit(f, x, c, '+')
        # Tính giới hạn của hàm khi x tiến đến c 
        limit_fx = sp.limit(f, x, c)
        # Kiểm tra tính liên tục
        #if f_c == limit_from_left == limit_from_right:
        if f_c == limit_fx:
            print("The function is continuous at x =",c, ".")
        else:
            print(f"The function is not continuous at x = {c}.")


def exercise6():
    def a():
        x = sp.symbols('x')
        f6a = (x*x - x - 6) / (x-3)
        #At point x = 0
        lm_x_0 = sp.limit(f6a, x, 0)
        if lm_x_0!= 5: #f(0)=5
            print(f"Compare lm_x_0 = {lm_x_0} and f(0) = 5") #f(0) = 5
        else:
            print("Not continuous at x = 0")






def exercise7():
    x = sp.symbols('x')
    f7a = (x**2 - x -2)/(x - 2) #biểu thức 7a 
    for c in np.arange(-100, 101, 1):
        if c != 0:
            lm = sp.limit(f7a, x, c)
        if lm != f7a.subs(x, 1):     # f7a.subs(x, c) == f(c)?
            print("Fx is not continuous at point x = ", c)
exercise7()