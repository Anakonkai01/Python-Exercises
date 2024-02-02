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
    def _6b_():
        x = np.linspace(-10, 10, 400)
        k_values = [2, 4, 6, 8, 10, 12]
        for k in k_values:
            y = (x + k)**2
            plt.plot(x, y, label=f'f(x) = (x + {k})^2')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Graphs of f(x) = (x + k)^2 for different k values')
        plt.grid(True)
        plt.legend()
        plt.show()

    def _6a_():
        x = np.linspace(-10, 10, 400)
        k_values = [2, 4, 6, 8, 10, 12]
        for k in k_values:
            y = x**2 + k
            plt.plot(x, y, label=f'f(x) = x^2 + {k}')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Graphs of f(x) = x^2 + k for different k values')
        plt.grid(True)
        plt.legend()
        plt.show()

    def _6c_():
        x = np.linspace(-1, 10, 400)
        k_values = [1/3, 1, 3, 6]
        for k in k_values:
            y = k * np.sqrt(x)
            plt.plot(x, y, label=f'f(x) = {k}*sqrt(x)')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Graphs of f(x) = {k}*sqrt(x) for different k values')
        plt.grid(True)
        plt.legend()
        plt.show()

    def _6d_():
        x = np.linspace(-2, 0, 100)
        f = lambda x: (x + 1)**3 - 1
        y = f(x)
        plt.plot(x, y, label='f(x) = (x + 1)^3 - 1')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Graph of f(x) = (x + 1)^3 - 1')
        plt.grid(True)
        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)
        plt.legend()
        plt.show()

    def _6e_():
        def f(x):
            return (x - 1)**(2/3) - 1
        x = np.linspace(1, 3, 100)
        y = f(x)
        plt.plot(x, y, label='f(x) = (x - 1)^(2/3) - 1')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Graph of f(x) = (x - 1)^(2/3) - 1')
        plt.grid(True)
        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)
        plt.legend()
        plt.show()

    def _6f_():
        def f(x):
            return 0.5 * (x + 1) + 5
        x = np.linspace(-10, 10, 100)
        y = f(x)
        plt.plot(x, y, label='f(x) = 0.5 * (x + 1) + 5')
        plt.xlabel('x-axis')
        plt.ylabel('y-axis')
        plt.title('Graph of f(x) = 0.5 * (x + 1) + 5')
        plt.grid(True)
        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)
        plt.legend()
        plt.show()

    def _6g_():
        def f(x):
            return 1 / x**2
        x = np.linspace(-10, -0.1, 1000)
        y = f(x)
        plt.plot(x, y, label='f(x) = 1 / x^2')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Graph of f(x) = 1 / x^2 with Left Shift and Down Shift')
        plt.grid(True)
        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)
        plt.legend()
        plt.show()

    def _6h_():
        def f(x):
            return 1 - x**3
        x = np.linspace(-2, 2, 100)
        y = f(x)
        plt.plot(x, y, label='Original f(x) = 1 - x^3')
        x_stretched = np.linspace(-4, 4, 100)
        y_stretched = f(x_stretched / 2)
        plt.plot(x_stretched, y_stretched, label='Stretched f(x) = 1 - (x/2)^3')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Graph of f(x) = 1 - x^3 and its Horizontal Stretch')
        plt.grid(True)
        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)
        plt.legend()
        plt.show()

    def _6i_():
        def f(x):
            return np.sqrt(x + 1)
        x = np.linspace(-1, 9, 100)
        y = f(x)
        plt.plot(x, y, label='Original f(x) = sqrt(x + 1)')
        x_compressed = np.linspace(-2, 18, 100)
        y_compressed = f(x_compressed*4)
        plt.plot(x_compressed, y_compressed, label='Compressed f(x) = sqrt(4x + 1)')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Graph of f(x) = sqrt(x + 1) and its Horizontal Compression')
        plt.grid(True)
        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)
        plt.legend()
        plt.show()

    def _6j_():
        def f(x):
            return np.sqrt(x + 1)
        x = np.linspace(-1, 9, 100)
        y = f(x)
        plt.plot(x, y, label='Original f(x) = sqrt(x + 1)')
        x_stretched = np.linspace(-1, 9, 100)
        y_stretched = 3 * f(x_stretched)
        plt.plot(x_stretched, y_stretched, label='Stretched f(x) = 3 * sqrt(x + 1)')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Graph of f(x) = sqrt(x + 1) and its Vertical Stretch')
        plt.grid(True)
        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)
        plt.legend()
        plt.show()

    functions_dict = {
        '6a': _6a_,
        '6b': _6b_,
        '6c': _6c_,
        '6d': _6d_,
        '6e': _6e_,
        '6f': _6f_,
        '6g': _6g_,
        '6h': _6h_,
        '6i': _6i_,
        '6j': _6j_
    }

    def plot_selected_function():
        while True:
            user_input = input("Enter the function you want to execute (6a, 6b, 6c, 6d, 6e, 6f, 6g, 6h, 6i, 6j) or ('no') to exit: ").lower()
            if user_input == 'no':
                break
            if user_input in functions_dict:
                selected_function = functions_dict[user_input]()
            else:
                print("Invalid input. Please enter a valid function.")

    plot_selected_function()


def exercise7():
    def is_one_to_one(func, domain):
        output_set = set()
        for input_val in domain:
            output_val = func(input_val)
            if output_val in output_set:
                return False
            output_set.add(output_val)
        return True

    def f1(x):
        return x**3 - x/2

    domain1 = range(-100, 101)
    print("Function f(x) = x^3 - x/2 is one-to-one:", is_one_to_one(f1, domain1))

    def f2(x):
        return x**2 + x/2

    domain2 = range(-100, 101)
    print("Function f(x) = x^2 + x/2 is one-to-one:", is_one_to_one(f2, domain2))





def choose():
    while True:
        user_input = input("Enter the number of the exercise you want to execute (1, 2, 3, 4, 5, 6, 7, NO to exit): ").lower()
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
        elif user_input == "7":
            exercise7()
        else:
            print("Invalid input. Please enter a valid exercise number or 'NO' to exit.")
        
choose()