import math 
import numpy as np
import matplotlib.pyplot as plt


# Exercise 1:
def exercise1():
    def a(x):
        return math.sqrt(x)

    def b(x):
        return x**(1/3)

    def c(x):
        return x**(2/3)

    def d(x):
        return (x**3/3) - (x**2/2) - 2*x + 1/3

    def e(x):
        return (2*x**2 - 3)/(7*x+4)

    def f(x):
        return (5*x**2 + 8*x - 3)/(3*x**2 + 2)

    def g(x):
        x = math.radians(x)
        return math.sin(x)

    def h(x):
        x = math.radians(x)
        return math.cos(x)

    def i(x):
        return 3**x

    def j(x):
        return 10**(-x)

    def k(x):
        return math.e**x

    def l(x):
        return math.log2(x)

    def m(x):
        return math.log10(x)

    def n(x):
        return math.log(x)

    functions_dict = {
        'a': a,
        'b': b,
        'c': c,
        'd': d,
        'e': e,
        'f': f,
        'g': g,
        'h': h,
        'i': i,
        'j': j,
        'k': k,
        'l': l,
        'm': m,
        'n': n,
    }

    def selected_function():
        while True:
            user_input = input("Enter the function code (a, b, c, d, e, f, g, h, i, j, k, l, m, n, or 'NO' to exit): ").lower()

            if user_input == 'no':
                break

            if user_input in functions_dict:
                selected_function = functions_dict[user_input]

                x = float(input("Enter the value of x: "))

                result = selected_function(x)
                print(f"Result of {user_input}({x}) = {result}")
            else:
                print("Invalid input. Please enter a valid function code or 'NO' to exit.")

    selected_function()



# Exercise 2:
def exercise2():
    def fx_2e(x):
        return x if x >= 0 else -x

    def fx_2a(x):
        return 2 + (x**2)/(x**2+4)

    def fx_2b(x):
        return math.sqrt(5*x + 10)

    def fx_2c(x):
        return 2/(x**2-16)

    def fx_2d(x):
        return x**4 + 3*x**2 - 1

    functions_dict = {
        '2e': fx_2e,
        '2a': fx_2a,
        '2b': fx_2b,
        '2c': fx_2c,
        '2d': fx_2d,
    }

    def find_max_min(func, range_start, range_end, step):
        arr = np.array([])
        for x in np.arange(range_start, range_end + step, step):
            arr = np.append(arr, round(func(x), 5))
        
        print("Max:", arr.max())
        print("Min:", arr.min())

    while True:
        user_input = input("Enter the function code (2e, 2a, 2b, 2c, 2d, NO to exit): ").lower()

        if user_input == "no":
            break  

        if user_input in functions_dict:
            selected_function = functions_dict[user_input]

            if user_input == "2e":
                range_start, range_end, step = -3.0, 3.0, 0.0001
            elif user_input == "2a":
                range_start, range_end, step = -2.0, 2.0, 0.0001
            elif user_input == "2b":
                range_start, range_end, step = 0.0, 5.0, 0.0001
            elif user_input == "2c":
                range_start, range_end, step = -5.0, 10.0, 0.0001
            elif user_input == "2d":
                range_start, range_end, step = -3.0, 3.0, 0.0001

            find_max_min(selected_function, range_start, range_end, step)
        else:
            print("Invalid input. Please enter a valid function code or 'NO' to exit.")



# Exercise 3:
def exercise3():
    f1 = lambda x:x + 5
    f2 = lambda x:x**2 - 3

    fCompositeA = f1(f2(0))
    fCompositeB = f2(f1(0))
    fCompositeC = f1(f2(-5))
    fCompositeD = f2(f2(2))

    functions_dict = {
        'a': fCompositeA,
        'b': fCompositeB,
        'c': fCompositeC,
        'd': fCompositeD
    }

    def selected_function():
        while True:
            user_input = input("Enter the function you want to execute (a,   b,   c,   d) or ('no') to exit: ").lower()
            if user_input == 'no':
                break
            if user_input in functions_dict:
                selected_function = functions_dict[user_input]
                print("The result = ",selected_function)

    selected_function()        


# exercise 4:
def exercise4():
    fx_4i = lambda x: x**(-3) if x != 0 else np.inf
    fx_4k = lambda x: -(1/x) if x != 0 else np.inf
    fx_4m = lambda x: math.sqrt(abs(x))
    fx_4j = lambda x: 1/math.pow(x, 2) if x != 0 else np.inf
    fx_4l = lambda x: 1/abs(x) if x != 0 else np.inf
    fx_4n = lambda x: math.sqrt(abs(-x))


    functions_dict = {
        '4i': fx_4i,
        '4k': fx_4k,
        '4m': fx_4m,
        '4j': fx_4j,
        '4l': fx_4l,
        '4n': fx_4n,
    }

    def plot_selected_function():

        while True:
            user_input = input("Enter the function you want to execute (4i,  4k,  4m,  4j,  4l,  4n) or ('no') to exit: ").lower()

            if user_input == 'no':
                break

            if user_input in functions_dict:
            
                selected_function = functions_dict[user_input]

                x_values = np.arange(-50.0, 50.1, 0.5)


                y_values = list(map(selected_function, x_values))

                plt.grid()
                plt.plot(x_values, y_values)
                plt.title(f"Plot of {user_input}")
                plt.show()
            else:
                print("Invalid input. Please enter a valid function.")

    plot_selected_function()






# exercise 5
def exercise5():
    def fx1(x):
        return math.sqrt(1 - math.pow((abs(x) - 1),2))

        
    def fx2(x):
        return -3 * math.sqrt(1 - math.sqrt(abs(x) / 2))

    x = np.arange(-2.0, 2.0, 0.0001)
    y1 = [fx1(xi) for xi in x]
    y2 = [fx2(xi) for xi in x]
    plt.plot(x, y1, color='magenta', label='fx1: Semi-circle')
    plt.plot(x, y2, color='red', label='fx2')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title('Plot of fx1 and fx2')
    plt.grid(True)
    plt.show()


# exercise 6:
def exercise6():
    def _6a_():
        x = np.linspace(-10, 10, 400)
        k_values = [2, 4, 6, 8, 10, 12]
        fig, ax = plt.subplots()
        for k in k_values:
            y = (x + k)**2
            ax.plot(x, y, label=f'f(x) = (x + {k})^2')
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.set_title('Graphs of f(x) = (x + k)^2 for different k values')
        ax.grid(True)
        ax.legend()
        plt.show()

    def _6b_():
        x = np.linspace(-10, 10, 400)
        k = [2, 4, 6, 8, 10, 12]

        fig, ax = plt.subplots()
        for k in k:
            y = x**2 + k
            ax.plot(x, y, label=f'f(x) = x^2 + {k}')
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.set_title('Graphs of f(x) = x^2 + k for different k values')
        ax.grid(True)
        ax.legend()
        plt.show()



def choose():
    while True:
        user_input = input("Enter the number of the exercise you want to execute (1, 2, 3, 4, 5, 6, NO to exit): ").lower()
        if user_input == "no":
            break
        if user_input == "1":
            exercise1()
        elif user_input == "2":
            exercise2()
        elif user_input == "3":
            exercise3()
        elif user_input == "4":
            exercise4()
        elif user_input == "5":
            exercise5()
        elif user_input == "6":
            exercise6()
        else:
            print("Invalid input. Please enter a valid exercise number or 'NO' to exit.")
        
choose()