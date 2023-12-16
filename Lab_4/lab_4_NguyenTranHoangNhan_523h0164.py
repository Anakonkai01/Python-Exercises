import sympy as sp 
import math  
import numpy as np 
import matplotlib.pyplot as plt
from sympy import AccumBounds



# exercise 1 
def exercise1():
    x = sp.symbols('x')
    functions = {
        'a1': lambda: sp.limit(sp.Abs(x**2 - x - 7), x, 3),
        'b1': lambda: sp.limit(sp.Abs(x - 1)/(x**2 - 1), x, 1),
        'c1': lambda: sp.limit(sp.exp(1.0)/x, x, 1),
        'd1': lambda: sp.limit((x**4 - 16)/(x - 2), x, 2),
        'e1': lambda: sp.limit((x**3 - x**2 - 5*x - 3)/((x - 1)**2), x, -1),
        'f1': lambda: sp.limit((x**2 - 9)/((x**2 + 7)**(1/2) - 4), x, 3),
        'g1': lambda: sp.limit((sp.Abs(x))/(sp.sin(sp.rad(x))), x, 1),
        'h1': lambda: sp.limit((1 - sp.cos(sp.rad(x)))/(x*sp.sin(sp.rad(x))), x, 0),
        'i1': lambda: sp.limit((2*x**2)/(3 - 3*sp.cos(sp.rad(x))), x, 0),
        'j1': lambda: sp.limit(((3 + x)/(-1 + x))**x, x, sp.oo),
        'k1': lambda: sp.limit((1 - (2)/(3 + x))**x, x, sp.oo),
        'l1': lambda: sp.limit((1/x)**(1/x), x, sp.oo),
        'm1': lambda: sp.limit((-(x**(1/3)) + (1 + x)**(1/3))/(-(x**(1/2)) + (1 + x)**(1/2)), x, sp.oo),
        'n1': lambda: sp.limit(sp.factorial(x)/(x**x), x, sp.oo),
    }

    while True:
        choice = input("Choose a function (a1, b1, c1, d1, e1, f1, g1, h1, i1, j1, k1, l1, m1, n1) or 'no' to exit:").lower()

        if choice == 'no':
            break

        if choice in functions:
            result = functions[choice]()
            print(result)
        else:
            print("Invalid choice. Try again.")


def exercise2():
    x = sp.symbols('x')
    def plot_and_limit(expression, limit_point):
        x_values = np.arange(limit_point - 2, limit_point + 2, 0.01)
        y_values = [expression.subs(x, val) if sp.im(expression.subs(x, val)) == 0 else None for val in x_values]

        plt.plot(x_values, y_values, label=str(expression))
        plt.scatter(limit_point, expression.subs(x, limit_point), color='red', marker='o', label=f'Limit Point ({limit_point}, {expression.subs(x, limit_point)})')

    def choose_and_plot(choice):
        if choice in functions:
            expression = functions[choice]['expression']
            limit_point = functions[choice]['limit_point']
            plot_and_limit(expression, limit_point)
            plt.legend()
            plt.title(f"Function: {choice}")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.show()
        else:
            print("Invalid choice. Try again.")

    # Define the functions and their limit points
    functions = {
        'a2': {'expression': sp.Abs(x**2 - x - 7), 'limit_point': sp.limit(sp.Abs(x**2 - x - 7), x, 3)},
        'b2': {'expression': sp.Abs(x - 1)/(x**2 - 1), 'limit_point': sp.limit(sp.Abs(x - 1)/(x**2 - 1), x, 1)},
        'c2': {'expression': sp.exp(1.0)/x, 'limit_point': sp.limit(sp.exp(1.0)/x, x, 1)},
        'd2': {'expression': (x**4 - 16)/(x - 2), 'limit_point': sp.limit((x**4 - 16)/(x - 2), x, 2)},
        'e2': {'expression': (x**3 - x**2 - 5*x - 3)/((x - 1)**2), 'limit_point': sp.limit((x**3 - x**2 - 5*x - 3)/((x - 1)**2), x, -1)},
        'f2': {'expression': (x**2 - 9)/((x**2 + 7)**(1/2) - 4), 'limit_point': sp.limit((x**2 - 9)/((x**2 + 7)**(1/2) - 4), x, 3)},
        'g2': {'expression': sp.Abs(x)/sp.sin(sp.rad(x)), 'limit_point': sp.limit(sp.Abs(x)/sp.sin(sp.rad(x)), x, 1)},
        'h2': {'expression': (1 - sp.cos(sp.rad(x)))/(x*sp.sin(sp.rad(x))), 'limit_point': sp.limit((1 - sp.cos(sp.rad(x)))/(x*sp.sin(sp.rad(x))), x, 0)},
        'i2': {'expression': (2*x**2)/(3 - 3*sp.cos(sp.rad(x))), 'limit_point': sp.limit((2*x**2)/(3 - 3*sp.cos(sp.rad(x))), x, 0)},
        'j2': {'expression': ((3 + x)/(-1 + x))**x, 'limit_point': sp.limit(((3 + x)/(-1 + x))**x, x, sp.oo)},
        'k2': {'expression': (1 - (2)/(3 + x))**x, 'limit_point': sp.limit((1 - (2)/(3 + x))**x, x, sp.oo)},
        'l2': {'expression': (1/x)**(1/x), 'limit_point': sp.limit((1/x)**(1/x), x, sp.oo)},
        'm2': {'expression': (-(x**(1/3)) + (1 + x)**(1/3))/(-(x**(1/2)) + (1 + x)**(1/2)), 'limit_point': sp.limit((-(x**(1/3)) + (1 + x)**(1/3))/(-(x**(1/2)) + (1 + x)**(1/2)), x, sp.oo)},
        'n2': {'expression': sp.factorial(x)/(x**x), 'limit_point': sp.limit(sp.factorial(x)/(x**x), x, sp.oo)},
    }
    sp.init_printing()
    while True:
        print("Choose a function:", end=" ")
        for key in functions:
            print(f"{key}", end=" ")
        print("\nOr enter 'no' to exit.")

        choice = input("Enter your choice: ").lower()

        if choice == 'no':
            break

        choose_and_plot(choice)  









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
            print("Limit does not exist")
    function_dict = {
        'a3': a,
        'b3': b,
    }
    while True:
        choose = input("Choose a3 , b3 or 'no' to exit:")
        if choose in function_dict:
            function_dict[choose]()
        if choose == 'no':
            break


def exercise4():
    x = sp.symbol('x')
    f = math.sin(1/math.radians(x))
    lmRight = sp.limit(f, x, 0, '+')
    
    if isinstance(lmRight, AccumBounds):
        print("ERROR: there is no limit from the right side")
    else:
        print("Limit from the right to the point =", lmRight)
    
    print("Limit from the left to the point = ",0)
    print("Limit at the point = ",0)







def exercise5():
    def a():
        x = sp.symbols('x') 
        f = x**2 - 7
        c = 1 
        f_c = f.subs(x, c) 
        limit_fx = sp.limit(f, x, c)
        if f_c == limit_fx:
            print(f"The function is continuous at x = {c}.")
        else:
            print(f"The function is not continuous at x = {c}.")
    def b():
        x = sp.symbols('x') 
        f = (x*2 - 3)**1/2
        c = 2
        f_c = f.subs(x, c) 
        limit_fx = sp.limit(f, x, c)
        if f_c == limit_fx:
            print(f"The function is continuous at x = {c}.")
        else:
            print(f"The function is not continuous at x = {c}.")
    function_dict = {
        'a5': a,
        'b5': b,
    }
    while True:
        choose = input("Choose a5, b5 or 'no' to exit:")
        if choose in function_dict:
            function_dict[choose]()
        if choose == 'no':
            break

def exercise6():
    def a():
        x = sp.symbols('x')
        f6a = (x*x - x - 6) / (x-3)
        #At point x = 0
        lm_x_0 = sp.limit(f6a, x, 0)
        if lm_x_0!= 5: #f(0)=5
            print("Not continuous at x = 0")
        else:
            print("The function is continuous at x = 0")
    def b():
        x = sp.symbols('x')
        f6b = (x**3 -8)/(x**2-4)
        #At point x = 0
        lm_x_2 = sp.limit(f6b, x, 2)
        lm_x_minus2 = sp.limit(f6b,x,-2)

        if lm_x_2!= 3: #f(0)=5
            print("Not continuous at x = 2")
        else:
            print("The function is continuous at x = 2")

        if lm_x_minus2!= 4: 
            print("Not continuous at x = -2")
        else:
            print("The function is continuous at x = -2")
    def c():
        x = sp.symbols('x')
        f6c = (x**2 - x -2 )/(x-2)
        lm_x_2 = sp.limit(f6c, x, 2)
        if lm_x_2 != 1:
            print("Not continuous at x = 2")
        else:
            print("The function is continuous at x = 2")
    def d():
        x = sp.symbols('x')
        f6d = 1/(x**2)
        lm_x_0 = sp.limit(f6d, x, 0)
        if lm_x_0!= 1:
            print("Not continuous at x = 0")
        else:
            print("The function is continuous at x = 0")

    function_dict = {
        'a6': a,
        'b6': b,
        'c6': c,
        'd6': d,
    }
    while True:
        choose = input("Choose a6, b6, c6, d6 or 'no' to exit:")
        if choose in function_dict:
            function_dict[choose]()
        if choose == 'no':
            break

    




def exercise7():
    x = sp.symbols('x')
    def a():
        f7a = (x**2 - x - 2)/(x - 2)  
        for c in np.arange(-100, 100, 1):
            lm = sp.limit(f7a, x, c)
            if lm != f7a.subs(x, c):   
                print("Fx is not continuous at point x = ", c)
    def b():
        f7b = (x**2-2*x -3)/(2*x-6)
        for c in np.arange(-100, 100, 1):
            lm = sp.limit(f7b, x, c)
            if lm!= f7b.subs(x, c):   
                print("Fx is not continuous at point x = ", c)
    function_dict = {
        'a7': a,
        'b7': b,
    }
    while True:
        choose = input("Choose a7, b7 or 'no' to exit:")
        if choose in function_dict:
            function_dict[choose]()
        if choose == 'no':
            break


def exercise8():
    x = sp.symbols('x')
    fx = 1 - sp.sqrt(1 - x**2)
    check = 0
    threshold = 1e-10  # Choose a small threshold

    for c in np.arange(-1, 1.1, 0.1):
        if abs(c) < threshold:
            continue

        try:
            lmRight = sp.limit(fx, x, c, dir='+')
            lmLeft = sp.limit(fx, x, c, dir='-')
            if sp.simplify(lmRight) != sp.simplify(lmLeft):
                print("Fx is not continuous at point x =", c)
                check += 1
            else:
                print("Fx is continuous at point x =", c)
        except NotImplementedError:
            print("Limit computation not implemented for x =", c)
            check += 1

    if check == 0:
        print("Fx is continuous at all points in the interval [ -1 ; 1 ]")



def exercise9():
    def function_y(x):
        return np.sin(10 * np.pi / x)

    def secant_line(x, x1=1):
        return np.sin(10 * np.pi / x1) / (x1 - 1) * (x - 1)

    # Values for x
    x_values = np.linspace(0.1, 2.5, 500)

    # Calculate y values for the function and secant line
    y_function = function_y(x_values)
    y_secant = secant_line(x_values)

    # Plot the function and secant line
    plt.figure(figsize=(8, 6))
    plt.plot(x_values, y_function, label='y = sin(10*pi/x)')
    plt.scatter([1], [0], color='red', marker='o', label='P(1, 0)')

    # Plot secant lines for different x values
    for x_val in [2, 1.5, 1.4, 1.3, 1.2, 1.1, 0.5, 0.6, 0.7, 0.8, 0.9]:
        y_secant_x = secant_line(x_values, x_val)
        label = f'Secant at x={x_val}'
        plt.plot(x_values, y_secant_x, label=label, linestyle='--')

        # Calculate the slope in radians and print the result
        slope_val_radians = round(np.sin(10 * np.pi / x_val) / (x_val - 1),2)
        print(f"Slope of secant at x={x_val} in radians: {slope_val_radians}")

        # Convert the slope to degrees
        slope_val_degrees = round(np.degrees(slope_val_radians),2)
        print(f"Slope of secant at x={x_val} in degrees: {slope_val_degrees}")

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Graph of y = sin(10*pi/x) and Secant Lines')
    plt.legend()
    plt.grid(True)
    plt.show()



def exercise10():
    x = sp.symbol('x')
    lm_x_0 = sp.limit(sp.sin(x)/x , x, 0)
    print("Fx = sin(x)/x  at point 0 =" , lm_x_0)
    lm_x_2 = sp.limit((x**2 + x - 6)/(x**2 - 4) , x , 2)
    print("Fx = (x^2 + x - 6)/(x^2- 4) at point 2 =", lm_x_2)






exercise_dict = {
    '1': exercise1,
    '2': exercise2,
    '3': exercise3,
    '4': exercise4,
    '5': exercise5,
    '6': exercise6,
    '7': exercise7,
    '8': exercise8,
    '9': exercise9,
    '10': exercise10,
}

while True:
    choose = input("Choose an exercise [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ]or 'no' to exit: ")
    if choose in exercise_dict:
        exercise_dict[choose]()
    if choose == 'no':
        break

    