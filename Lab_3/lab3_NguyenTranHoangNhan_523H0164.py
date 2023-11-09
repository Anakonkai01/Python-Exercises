import math 
import numpy as np
import matplotlib.pyplot as plt


# Exercise 1:
def exercise1():
    x = float(input("Input for value of x"))
    # a
    def a(x):
        cal = math.pow(x,(1/2))
        print(f"sqrt of x = {cal}")
    # b
    def b(x):
        cal = math.pow(x,(1/3))
        print(f"cube root of x = {cal}")
    # c 
    def c(x):
        cal = math.pow(x,(2/3))
        print(f"x power 2/3 = {cal}")
    # d
    def d(x):
        cal = (x**3/3) - (x**2/2) - 2*x + 1/3
        print(f"Result of (x**3/3) - (x**2/2) - 2*x + 1/3 = {cal}")
    # e
    def e(x):
        cal = (2*x**2 - 3)/(7*x+4)
        print(f"Result of  = (2*x**2 - 3)/(7*x+4) = {cal}")
    # f
    def f(x):
        cal = (5*x**2 + 8*x - 3 )/(3*x**2 + 2)
        print(f"Result of (5*x**2 + 8*x - 3 )/(3*x**2 + 2) = {cal}")
    # g
    def g(x):
        x = math.radians(x)
        cal = math.sin(x)
        print(f"sin({x}) = {cal}") 
    # h
    def h(x):
        x = math.radians(x)
        cal = math.cos(x)
        print(f"cos({x}) = {cal}")
    # i 
    def i(x):
        cal = math.pow(3,x)
        print(f"3**x = {cal}")
    # j
    def j(x):
        cal = math.pow(10,-x)
        print(f"10**-x = {cal}")
    # k
    def k(x):
        cal = math.pow(math.e,x)
        print(f"e**x = {cal}")
    # l
    def l(x):
        cal = math.log2(x)
        print(f"Log(x,base = 2) = {cal}")
    # m 
    def m(x):
        cal = math.log10(x)
        print(f"log(x,base = 10) = {cal}")
    # n
    def n(x):
        cal = math.log(x)
        print(f"ln(x)= {cal}")
    # # default
    # def default_case():
    #     print('Invalid Input')

    # def switch_case(value):
    #     cases = {
    #         'a': a(x),
    #         'b': b(x),
    #         'c': c(x),
    #         'd': d(x),
    #         'e': e(x),
    #         'f': f(x),
    #         'g': g(x),
    #         'h': h(x),
    #         'i': i(x),
    #         'j': j(x),
    #         'k': k(x),
    #         'l': l(x),
    #         'm': m(x),
    #         'n': n(x),
    #     }
    #     case = cases.get(value, default_case)
    #     case()
    #     return cases

    # value = input(f"Input")
    # switch_case(value)

# End of Exercise 1:


# Exercise 2:
def exercise2():
    # 2e
    def fx_2e(x):
        if x >= 0:
            return x
        else:
            return -x
    # 2a
    def fx_2a(x):
        cal = 2 + (x**2)/(x**2+4)
        return cal
    # 2b
    def fx_2b(x):
        cal = math.sqrt(5*x +10)
        return cal
    # 2c
    def fx_2c(x):
        cal = 2/(x**2-16)
        return cal
    # 2d
    def fx_2d(x):
        cal = x**4 + 3*x**2 - 1 
        return cal


    # Find The Max and Min Value
    def exer2e():
        arr = np.array([])
        for x in np.arange( -3.0, 3.1, 0.0001 ):
            arr = np.append(arr,round(fx_2e(x),5))
        
        print(arr.max())
        print(arr.min())


    def exer2a():
        arr = np.array([])
        for x in np.arange( -2.0, 2.1, 0.0001 ):
            arr = np.append(arr,round(fx_2a(x),5))
        
        print(arr.max())
        print(arr.min())


    def exer2b():
        arr = np.array([])
        for x in np.arange( -0.0, 5.1, 0.0001 ):
            arr = np.append(arr,round(fx_2b(x),5))
        
        print(arr.max())
        print(arr.min())


    def exer2c():
        arr = np.array([])
        for x in np.arange( -5.0, 10.1, 0.0001 ):
            arr = np.append(arr,round(fx_2c(x),5))
        
        print(arr.max())
        print(arr.min())


    def exer2d():
        arr = np.array([])
        for x in np.arange( -3.0, 3.1, 0.0001 ):
            arr = np.append(arr,round(fx_2d(x),5))
        
        print(arr.max())
        print(arr.min())
# End of excercise 2


# Exercise 3:
def exercise3():
    f1 = lambda x:x + 5
    f2 = lambda x:x**2 - 3

    fCompositeA = f1(f2(0))
    fCompositeB = f2(f1(0))
    fCompositeC = f1(f2(-5))
    fCompositeD = f2(f2(2))


# exercise 4:
def exercise4():
    fx_4i = lambda x:x**(-3)
    fx_4k = lambda x: -(1/(x))
    x2 = np.arange(-50.0, 50.1, 0.5)
    x1 = np.arange(-20.0, 20.1, 0.5)
    y1 = list(map(fx_4i, x1))
    y2 = list(map(fx_4k,x2))
    choose = int(input("Input '1' for fx_4i '2' for fx_4k "))
    plt.grid()
    if choose == 1:
        plt.plot(x1,y1)
    else :
        plt.plot(x2,y2)   
    plt.show()


# exercise 5
def exercise5():
    def fx1(x):
        if abs(x) <= 1:
            return math.sqrt(1 - (abs(x) - 1) ** 2)
        else:
            return 0.0  # Return 0 for x values outside the valid range
        
    def fx2(x):
        if abs(x) <= 4:
            return -3 * math.sqrt(1 - math.sqrt(abs(x) / 2))
        else:
            return 0.0  # Return 0 for x values outside the valid range
    x = np.arange(-2.0, 2.0, 0.01)
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
