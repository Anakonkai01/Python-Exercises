import numpy as np
from scipy.optimize import minimize_scalar
import sympy as sp
import matplotlib.pyplot as plt

def exercise1():
    print("Exercise 1:")
    x = sp.symbols('x')

    fa = 3*x**4 + 16*x**3 + 18*x**2 - 9
    fb = (x+2)/(2*x**2)
    fc = -x**2/3 + x**2 + 3*x + 4
    fd = (5*x**2 + 5)/x

    dfa = sp.diff(fa,x)
    dfb = sp.diff(fb,x)
    dfc = sp.diff(fc,x)
    dfd = sp.diff(fd,x)

    xa = sp.solve(sp.Eq(dfa,0),x)
    xb = sp.solve(sp.Eq(dfb,0),x)
    xc = sp.solve(sp.Eq(dfc,0),x)
    xd = sp.solve(sp.Eq(dfd,0),x)
    print(f"Exercise 1a: f(x) has {len(xa)} critical numbers: ",end=" ")
    for x_value in xa:
        y_value = fa.subs({x:x_value})
        print(f"({x_value},{y_value})",end=" ")
    print()
    print(f"Exercise 1b: f(x) has {len(xb)} critical numbers: ",end=" ")
    for x_value in xb:
        y_value = fb.subs({x:x_value})
        print(f"({x_value},{y_value})",end=" ")
    print()
    print(f"Exercise 1c: f(x) has {len(xc)} critical numbers: ",end=" ")
    for x_value in xc:
        y_value = fc.subs({x:x_value})
        print(f"({x_value},{y_value})",end=" ")
    print()
    print(f"Exercise 1d: f(x) has {len(xd)} critical numbers: ",end=" ")
    for x_value in xd:
        y_value = fd.subs({x:x_value})
        print(f"({x_value},{y_value})",end=" ")

def exercise2():
    print()
    print()
    print("Exercise 2:")
    x = sp.symbols('x')

    fa = 3*x**4 + 16*x**3 + 18*x**2 - 9
    fb = (x+2)/(2*x**2)
    fc = -x**2/3 + x**2 + 3*x + 4
    fd = (5*x**2 + 5)/x

    dfa = sp.diff(fa,x)
    dfb = sp.diff(fb,x)
    dfc = sp.diff(fc,x)
    dfd = sp.diff(fd,x)

    xa = sp.solve(sp.Eq(dfa,0),x)
    xb = sp.solve(sp.Eq(dfb,0),x)
    xc = sp.solve(sp.Eq(dfc,0),x)
    xd = sp.solve(sp.Eq(dfd,0),x)

    d2fa_c = sp.diff(fa,x,2)
    d2fb_c = sp.diff(fb,x,2)
    d2fc_c = sp.diff(fc,x,2)
    d2fd_c = sp.diff(fd,x,2)

    local_min = []
    local_max = []
    for x_value in xa:
        test = d2fa_c.subs({x:x_value})
        if(test > 0 ):
            local_min.append(x_value)
        elif(test < 0):
            local_max.append(x_value)
    if local_min :
        print("Exercise 2a: f(x) has local minimum at:",end=" ")
        for min_val in local_min:
            print(f" c = {min_val}",end=",")
    else:
        print("Exercise 2a: f(x) does not have local minimum",end=" ")
    print()
    if local_max :
        print("Exercise 2a: f(x) has local maximum at:",end=" ")
        for max_val in local_max:
            print(f" c = {max_val}",end=" ")
    else:
        print("Exercise 2a: f(x) does not have local maximum",end=" ")
    
    print()

    local_min.clear()
    local_max.clear()
    for x_value in xb:
        test = d2fb_c.subs({x:x_value})
        if(test > 0 ):
            local_min.append(x_value)
        elif(test < 0):
            local_max.append(x_value)
    print()
    if local_min:
        print("Exercise 2b: f(x) has local minimum at:",end=" ")
        for min_val in local_min:
            print(f" c = {min_val}",end=",")
    else:
        print("Exercise 2b: f(x) does not have local minimum",end=" ")
    print()
    if local_max:
        print("Exercise 2b: f(x) has local maximum at:",end=" ")
        for max_val in local_max:
            print(f" c = {max_val}",end=" ")
    else :
        print("Exercise 2b: f(x) does not have local maximum",end=" ")


    print()
    local_min.clear()
    local_max.clear()
    for x_value in xc:
        test = d2fc_c.subs({x:x_value})
        if(test > 0 ):
            local_min.append(x_value)
        elif(test < 0):
            local_max.append(x_value)
    print()
    if local_min:
        print("Exercise 2c: f(x) has local minimum at:",end=" ")
        for min_val in local_min:
            print(f" c = {min_val}",end=",")
    else:
        print("Exercise 2c: f(x) does not have local minimum",end=" ")
    print()
    if local_max:
        print("Exercise 2c: f(x) has local maximum at:",end=" ")
        for max_val in local_max:
            print(f" c = {max_val}",end=" ")
    else :
        print("Exercise 2c: f(x) does not have local maximum",end=" ")
    

    print()
    local_min.clear()
    local_max.clear()
    for x_value in xd:
        test = d2fd_c.subs({x:x_value})
        if(test > 0 ):
            local_min.append(x_value)
        elif(test < 0):
            local_max.append(x_value)
    print()
    if local_min:
        print("Exercise 2d: f(x) has local minimum at:",end=" ")
        for min_val in local_min:
            print(f" c = {min_val}",end=",")
    else:
        print("Exercise 2d: f(x) does not have local minimum",end=" ")
    print()
    if local_max:
        print("Exercise 2d: f(x) has local maximum at:",end=" ")
        for max_val in local_max:
            print(f" c = {max_val}",end=" ")
    else :
        print("Exercise 2d: f(x) does not have local maximum",end=" ")



def exercise3():
    print()
    print()
    print("Exercise 3:")
    x = sp.symbols('x')

    fa = x**3 - 27*x
    fb = (3/2)*x**4 - 4*x**3 + 4
    fc = (1/2)*x**4 - 4*x**2 + 5
    fd = (5/2)*x**4 - (20/3)*x**3 + 6

    dfa = sp.diff(fa,x)
    dfb = sp.diff(fb,x)
    dfc = sp.diff(fc,x)
    dfd = sp.diff(fd,x)

    xa = sp.solve(sp.Eq(dfa,0),x)
    xb = sp.solve(sp.Eq(dfb,0),x)
    xc = sp.solve(sp.Eq(dfc,0),x)
    xd = sp.solve(sp.Eq(dfd,0),x)

    for x_value in xa:
        if x_value < 0 or x_value > 5: xa.remove(x_value)
    xa.append(0)
    xa.append(5)

    for x_value in xb:
        if x_value < 0 or x_value > 3: xb.remove(x_value)
    xb.append(0)
    xb.append(3)

    for x_value in xc:
        if x_value < 1 or x_value > 3: xc.remove(x_value)
    xa.append(1)
    xa.append(3)

    for x_value in xd:
        if x_value < -1 or x_value > 3: xd.remove(x_value)
    xa.append(-1)
    xa.append(3)

    # xoa nhung gia tri bi trung lap bang tap hop set xong convert ve list
    xa =list(set(xa))
    xb =list(set(xb))
    xc =list(set(xc))
    xd =list(set(xd))

    ya = [fa.subs(x,x_value) for x_value in xa]
    yb = [fb.subs(x,x_value) for x_value in xb]
    yc = [fc.subs(x,x_value) for x_value in xc]
    yd = [fd.subs(x,x_value) for x_value in xd]

    print(f"Exercise 3a: The absolute maximum f(x) is: {max(ya)}")
    print(f"Exercise 3a: The absolute minimum f(x) is: {min(ya)}")
    print()
    print(f"Exercise 3b: The absolute maximum f(x) is: {max(yb)}")
    print(f"Exercise 3b: The absolute minimum f(x) is: {min(yb)}")
    print()
    print(f"Exercise 3c: The absolute maximum f(x) is: {max(yc)}")
    print(f"Exercise 3c: The absolute minimum f(x) is: {min(yc)}")
    print()
    print(f"Exercise 3d: The absolute maximum f(x) is: {max(yd)}")
    print(f"Exercise 3d: The absolute minimum f(x) is: {min(yd)}")

def exercise4():
    print()
    print()
    print("Exercise 4:")

    x = sp.symbols('x')
    functions_intervals = [
        (x**2 - 2*x - 5, (0, 2)),           # fa
        (3*x + x**3 + 5, (-4, 4)),          # fb
        (sp.sin(x) + 3*x**2, (-2, 2)),      # fc
        (sp.E**(x*2) + 3*x, (-1, 1)),       # fd
        (x**3 - 3*x, (-3, 0)),              # fe
        (x**3 - 3*x, (0, 3)),               # ff
        (sp.sin(x), (0, np.pi)),            # fg
        (sp.sin(2*x), (0, 2)),              # fh
        (sp.cos(x), (sp.pi/2, 3*sp.pi/2)),  # fi
        (sp.tan(x)**2, (-sp.pi/4, sp.pi/4)),# fj
        (sp.E**x*sp.sin(x), (0, np.pi)),    # fk
        (x**4 - 3*x**2, (-4, 0)),           # fl
        (x**4 - 3*x**2, (0, 4)),            # fm
        (x**5 - 5*x**3, (-4, 0)),           # fn
        (x**6 - 5*x**2, (-1, 1)),           # fo
        (x**3 - 9*x, (-3, 0)),              # fp
        (x**3 - 9*x, (0, 3)),               # fq
        (x**3 + 9*x, (-1, 1))               # fr
    ]

    for function, interval in functions_intervals:
        func = sp.lambdify(x, function, 'numpy')

        try:
            result_min = minimize_scalar(func, bounds=interval, method='bounded')

            result_max = minimize_scalar(lambda x: -func(x), bounds=interval, method='bounded')  

            optimal_x_min = result_min.x
            optimal_y_min = result_min.fun
            optimal_x_max = result_max.x
            optimal_y_max = -result_max.fun  

            if (
                np.isscalar(optimal_x_min) and np.isfinite(optimal_x_min) and np.isscalar(optimal_y_min) and np.isfinite(optimal_y_min) and
                np.isscalar(optimal_x_max) and np.isfinite(optimal_x_max) and np.isscalar(optimal_y_max) and np.isfinite(optimal_y_max)
            ):
                print(f"Function: {function}")
                print("Minimum Point: x =", optimal_x_min, ", y =", optimal_y_min)
                print("Maximum Point: x =", optimal_x_max, ", y =", optimal_y_max)

                x_values = np.linspace(interval[0], interval[1], 100)
                y_values = func(x_values)

                plt.plot(x_values, y_values, label=f"Function: {str(function)}")
                plt.scatter(optimal_x_min, optimal_y_min, color='red', label=f'Minimum Point at x = {optimal_x_min:.2f}', zorder=5)
                plt.scatter(optimal_x_max, optimal_y_max, color='green', label=f'Maximum Point at x = {optimal_x_max:.2f}', zorder=5)
                plt.title("Function with Minimum and Maximum Points")
                plt.xlabel("x")
                plt.ylabel("y")
                plt.legend()
                plt.grid(True)
                plt.show()
                print("\n" + "="*40 + "\n")
            else:
                print(f"Optimal values are not valid for function: {function}\n")

        except Exception as e:
            print(f"Error occurred while processing function {function}: {str(e)}\n")

    
def exercise5():
    print()
    print()
    print("Exercise 5:")
    print("Golden Search:")
    def golden_section_search(func, a, b, epsilon=1e-6):
        golden_ratio = (np.sqrt(5) - 1) / 2

        x1 = b - golden_ratio * (b - a)
        x2 = a + golden_ratio * (b - a)

        f_x1 = func(x1)
        f_x2 = func(x2)

        iteration = 1

        results = []

        while abs(b - a) > epsilon:
            results.append([iteration, a, b, x1, x2, f_x1, f_x2])

            if f_x1 < f_x2:
                b = x2
                x2 = x1
                x1 = b - golden_ratio * (b - a)
                f_x2 = f_x1
                f_x1 = func(x1)
            else:
                a = x1
                x1 = x2
                x2 = a + golden_ratio * (b - a)
                f_x1 = f_x2
                f_x2 = func(x2)

            iteration += 1

        results.append([iteration, a, b, x1, x2, f_x1, f_x2])

        return results

    def objective_function(x):
        return x**2

    a = -2
    b = 1
    epsilon = 0.3

    results = golden_section_search(objective_function, a, b, epsilon)

    print("Iteration |   a   |   b   |   x1   |   x2   |  f(x1)  |  f(x2) ")
    print("="*66)
    for row in results:
        print("{:9} | {:5.2f} | {:5.2f} | {:6.2f} | {:6.2f} | {:7.2f} | {:7.2f}".format(*row))

    x_values = np.linspace(a, b, 100)
    y_values = objective_function(x_values)
    plt.plot(x_values, y_values, label="f(x) = x^2")
    plt.scatter([row[3] for row in results], [row[5] for row in results], color='red', label='Iteration Points', zorder=5)
    plt.title("Golden Section Search for Minimum of f(x) = x^2")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True)
    plt.show()

def exercise6():
    print()
    print()
    print("Exercise 6:")
    print("Fibonacci Search:")
    def fibonacci_search(func, a, b, epsilon=1e-6):
        fib_sequence = [1, 1]
        while fib_sequence[-1] < (b - a) / epsilon:
            fib_sequence.append(fib_sequence[-1] + fib_sequence[-2])

        x1 = a + (b - a) * fib_sequence[-3] / fib_sequence[-1]
        x2 = a + (b - a) * fib_sequence[-2] / fib_sequence[-1]

        f_x1 = func(x1)
        f_x2 = func(x2)

        iteration = 1

        results = []

        while abs(b - a) > epsilon:
            results.append([iteration, a, b, x1, x2, f_x1, f_x2])

            if f_x1 < f_x2:
                b = x2
                x2 = x1
                x1 = a + (b - a) * fib_sequence[-3] / fib_sequence[-1]
                f_x2 = f_x1
                f_x1 = func(x1)
            else:
                a = x1
                x1 = x2
                x2 = a + (b - a) * fib_sequence[-2] / fib_sequence[-1]
                f_x1 = f_x2
                f_x2 = func(x2)

            iteration += 1

        results.append([iteration, a, b, x1, x2, f_x1, f_x2])

        return results

    def objective_function(x):
        return x**2

    a = -2
    b = 1
    epsilon = 0.3

    results = fibonacci_search(objective_function, a, b, epsilon)

    print("Iteration |   a   |   b   |   x1   |   x2   |  f(x1)  |  f(x2) ")
    print("="*66)
    for row in results:
        print("{:9} | {:5.2f} | {:5.2f} | {:6.2f} | {:6.2f} | {:7.2f} | {:7.2f}".format(*row))

    x_values = np.linspace(a, b, 100)
    y_values = objective_function(x_values)

    plt.plot(x_values, y_values, label="f(x) = x^2")
    plt.scatter([row[3] for row in results], [row[5] for row in results], color='red', label='Iteration Points', zorder=5)
    plt.title("Fibonacci Search for Minimum of f(x) = x^2")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True)
    plt.show()

def exercise7():
    print()
    print()
    print("Exercise 7:")
    x, m = sp.symbols('x m')

    y = x**3 - 3*m*x**2 + 3*(m**2 - 1)*x - (m**2 - 1)

    dy_dx = sp.diff(y, x)

    critical_points = sp.solve(dy_dx, x)

    m_values = []

    for point in critical_points:
        m_solution = sp.solve(y.subs(x, 1).subs(x, point), m)
        m_values.append(m_solution)

    for i, point in enumerate(critical_points):
        print(f"Solution {i + 1}:")
        print(f"   Critical Point x: {point}")
        print(f"   Corresponding m: {m_values[i]}\n")


exercise1()
exercise2()
exercise3()
exercise4()
exercise5()
exercise6()
exercise7()