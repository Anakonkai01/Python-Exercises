import sympy as sp 
import numpy as np
import matplotlib.pyplot as plt 


def exercise1():
    print()
    print()
    print("Exercise 1:")
    x = sp.symbols('x')

    fa = x**3 + 2*x**2 + 3
    fb = 1/x**3 + 1/x**2 + x*x**1/2
    fc = (x**3 + x*x**1/2 + x)/x**2
    fd = 2/x + x**3
    fe = x**2*(1/x + 2*x)
    ff = (x**1/2 - 1 )*(x + x**1/2 + 1)
    fg = (1 - 2/(sp.sin(x)**2))
    fh = 1/(sp.sin(x)**2*sp.cos(x)**2)
    fi = sp.E**x*(1-(sp.E**(-x))/(sp.cos(x)**2))
    fj = sp.E**x*(2 + sp.E**(-x)/sp.E**x)
    fk = (2**x + 2/x)
    fl = x**2*(x-1)**2
    fm = 1/(x*(x+1))
    fn = abs(1-x)
    fo = abs(2*x - x**2)
    fp = (x**2 - 3*x + 2)**1/2
    fq = (1 + sp.cos(2*x))**1/2


    dia = sp.integrate(fa,(x,1,2))
    dib = sp.integrate(fb,(x,1,4)) 
    dic = sp.integrate(fc,(x,1,4))
    did = sp.integrate(fd,(x,1,2))
    die = sp.integrate(fe,(x,1,2))
    dif = sp.integrate(ff,(x,0,1))
    dig = sp.integrate(fg,(x,sp.pi/4,sp.pi/2))
    dih = sp.integrate(fh,(x,sp.pi/6,sp.pi/4))
    dii = sp.integrate(fi,(x,0,sp.pi/4))
    dij = sp.integrate(fj,(x,0,sp.ln(2)))
    dik = sp.integrate(fk,(x,1,2))
    dil = sp.integrate(fl,(x,0,1))
    dim = sp.integrate(fm,(x,1,2))
    din = sp.integrate(fn,(x,0,2))
    dio = sp.integrate(fo,(x,0,3))
    dip = sp.integrate(fp,(x,2,4))
    diq = sp.integrate(fq,(x,0,sp.pi))


    print(f"Integral of fa over the interval [1, 2]: {dia}")
    print(f"Integral of fb over the interval [1, 4]: {dib}")
    print(f"Integral of fc over the interval [1, 4]: {dic}")
    print(f"Integral of fd over the interval [1, 2]: {did}")
    print(f"Integral of fe over the interval [1, 2]: {die}")
    print(f"Integral of ff over the interval [0, 1]: {dif}")
    print(f"Integral of fg over the interval [pi/4, pi/2]: {dig}")
    print(f"Integral of fh over the interval [pi/6, pi/4]: {dih}")
    print(f"Integral of fi over the interval [0, pi/4]: {dii}")
    print(f"Integral of fj over the interval [0, ln(2)]: {dij}")
    print(f"Integral of fk over the interval [1, 2]: {dik}")
    print(f"Integral of fl over the interval [0, 1]: {dil}")
    print(f"Integral of fm over the interval [1, 2]: {dim}")
    print(f"Integral of fn over the interval [0, 2]: {din}")
    print(f"Integral of fo over the interval [0, 3]: {dio}")
    print(f"Integral of fp over the interval [2, 4]: {dip}")
    print(f"Integral of fq over the interval [0, pi]: {diq}")


def exercise2():
    print()
    print()
    print("Exercise 2:")
    x,y = sp.symbols('x y')

    fa = x**3 -3*sp.sin(x)*sp.cos(x)
    fb = sp.sin(x**2)**2 
    fc = (1 + (x**2+1) + (x+1)**2)**1/2
    fd = x**2*y

    dia = sp.integrate(fa,(x,0,sp.pi/2))
    dib = sp.integrate(fb,(x,0,1))
    dic = sp.integrate(fc,(x,0,3))
    did1 = sp.integrate(fd,(x,0,3))
    did2 = sp.integrate(did1,(x,1,2))

    print(f"Integral of fa over the interval [0,pi/2]: {dia}")
    print(f"Integral of fb over the interval [0,1]: {dib}")
    print(f"Integral of fc over the interval [0,3]: {dic}")
    print(f"Integral over interval [1,2] of integral fd over the interval [1,2]: {did2}")

    fa_func = sp.lambdify(x, fa, 'numpy')
    fb_func = sp.lambdify(x, fb, 'numpy')
    fc_func = sp.lambdify(x, fc, 'numpy')
    fd_func = sp.lambdify(x, fd.subs(y, 2), 'numpy')

    # Generate x values
    x_values = np.linspace(0, 3, 100)

    # Calculate corresponding y values using the functions
    fa_values = fa_func(x_values)
    fb_values = fb_func(x_values)
    fc_values = fc_func(x_values)
    fd_values = fd_func(x_values)

    # Plot the graphs
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(x_values, fa_values, label='x**3 - 3*sin(x)*cos(x)')
    plt.title('Graph of x**3 - 3*sin(x)*cos(x) over [0, pi/2]')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(x_values, fb_values, label='sin(x**2)**2')
    plt.title('Graph of sin(x**2)**2 over [0, 1]')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(x_values, fc_values, label='sqrt(1 + (x**2 + 1) + (x + 1)**2)')
    plt.title('Graph of sqrt(1 + (x**2 + 1) + (x + 1)**2) over [0, 3]')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(x_values, fd_values, label='x**2*y (y=2)')
    plt.title('Graph of x**2*y (y=2) over [0, 3]')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()

    plt.tight_layout()
    plt.show()

def exercise3():
    print()
    print()
    print("Exercise 3:")

    x = sp.symbols('x')

    fa = x**2 -1 # [0,sqrt(3)]
    fb = -x**2/2 # [0,3]
    fc = -3*x**2 - 1 # [0,1]
    fd = x**2 - x # [-2,1]

    interval_a = (0, 3**1/2)
    interval_b = (0, 3)
    interval_c = (0, 1)
    interval_d = (-2, 1)

    fa_func = sp.lambdify(x, fa, 'numpy')
    fb_func = sp.lambdify(x, fb, 'numpy')
    fc_func = sp.lambdify(x, fc, 'numpy')
    fd_func = sp.lambdify(x, fd, 'numpy')

    x_values_a = np.linspace(interval_a[0], interval_a[1], 100)
    x_values_b = np.linspace(interval_b[0], interval_b[1], 100)
    x_values_c = np.linspace(interval_c[0], interval_c[1], 100)
    x_values_d = np.linspace(interval_d[0], interval_d[1], 100)

    fa_values = fa_func(x_values_a)
    fb_values = fb_func(x_values_b)
    fc_values = fc_func(x_values_c)
    fd_values = fd_func(x_values_d)

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(x_values_a, fa_values, label='x**2 - 1')
    plt.title('Graph of x**2 - 1 over [0, sqrt(3)]')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(x_values_b, fb_values, label='-x**2/2')
    plt.title('Graph of -x**2/2 over [0, 3]')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(x_values_c, fc_values, label='-3x**2 - 1')
    plt.title('Graph of -3x**2 - 1 over [0, 1]')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(x_values_d, fd_values, label='x**2 - x')
    plt.title('Graph of x**2 - x over [-2, 1]')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()

    plt.tight_layout()
    plt.show()

    average_a = sp.integrate(fa, (x, interval_a[0], interval_a[1])) / (interval_a[1] - interval_a[0])
    average_b = sp.integrate(fb, (x, interval_b[0], interval_b[1])) / (interval_b[1] - interval_b[0])
    average_c = sp.integrate(fc, (x, interval_c[0], interval_c[1])) / (interval_c[1] - interval_c[0])
    average_d = sp.integrate(fd, (x, interval_d[0], interval_d[1])) / (interval_d[1] - interval_d[0])

    print(f"Average value of fa(x) over [0, sqrt(3)]: {average_a.evalf()}")
    print(f"Average value of fb(x) over [0, 3]: {average_b.evalf()}")
    print(f"Average value of fc(x) over [0, 1]: {average_c.evalf()}")
    print(f"Average value of fd(x) over [-2, 1]: {average_d.evalf()}")


def exercise4():
    print()
    print()
    print("Exercise 4:")

    x = sp.symbols('x')
    f = x**2 * sp.cos(x)

    interval = (-4, 9)
    f_func = sp.lambdify(x, f, 'numpy')

    x_values = np.linspace(interval[0], interval[1], 1000)
    f_values = f_func(x_values)

    plt.figure(figsize=(8, 6))
    plt.plot(x_values, f_values, label='x**2*cos(x)')
    plt.fill_between(x_values, f_values, color='red', alpha=0.4, label='Area under the curve')
    plt.title('Function: x**2*cos(x) over [-4, 9] with Area under the curve')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()

    area = sp.integrate(f, (x, interval[0], interval[1]))
    print(f"Area under the curve of x**2*cos(x) over [-4, 9]: {area.evalf()}")


    f = sp.E**(-1/2*x**2)
    f_func = sp.lambdify(x, f, 'numpy')
    x_values = np.linspace(-5, 5, 1000)
    f_values = f_func(x_values)
    plt.figure(figsize=(8, 6))
    plt.plot(x_values, f_values, label='e**(-1/2*x**2)')
    plt.fill_between(x_values, f_values, color='lightcoral', alpha=0.4, label='Area under the curve')
    plt.title('Function: e**(1/2 * x**2) over [-4, 4] with Area under the curve')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()

    area_within_interval = sp.integrate(f, (x, -sp.oo, sp.oo))
    print(f"Area under the curve of e**(-1/2*x**2) over [-4, 4]: {area_within_interval.evalf()}")


def exercise5():
    print()
    print()
    print("Exercise 5:")
    t = sp.symbols('t')
    v = 160 - 32*t

    displacement = sp.integrate(v, (t, 0, 8))

    print(f"The displacement of the rock during the time period 0 ≤ t ≤ 8 is: {displacement.evalf()} feet.")


def exercise6():
    print()
    print()
    print("Exercise 6:")
    x = sp.symbols('x')
    marginal_cost = 1 / (2 * sp.sqrt(x))

    cost_function = sp.integrate(marginal_cost, x)

    cost_at_100 = cost_function.subs(x, 100)
    cost_at_1 = cost_function.subs(x, 1)
    total_cost = cost_at_100 - cost_at_1

    print(f"The cost of printing posters 2-100 is: {total_cost.evalf()} dollars.")


def exercise7():
    print()
    print()
    print("Exercise 7:")
    t = sp.symbols('t')
    H = (1 + t)**(1/2) + 5*t**(1/3)

    height_at_0 = H.subs(t, 0)
    height_at_4 = H.subs(t, 4)
    height_at_8 = H.subs(t, 8)

    print("Exercise 7a:")
    print(f"The height of the tree when t = 0: {height_at_0.evalf()} units.")
    print(f"The height of the tree when t = 4: {height_at_4.evalf()} units.")
    print(f"The height of the tree when t = 8: {height_at_8.evalf()} units.")


    a = 0
    b = 8

    print(f"Exercise 7b:")
    average_height = sp.integrate(H, (t, a, b)) / (b - a)
    print(f"Average height of the tree for 0 ≤ t ≤ 8: {average_height.evalf()} units.")


def exercise8():
    print()
    print()
    print("Exercise 8:")
    x = sp.symbols('x')
    def func_a(x):
        return 1 - x

    def func_b(x):
        return x**2 + 1

    def func_c(x):
        return np.cos(x)

    def func_d(x):
        return np.abs(x)

    x_values = np.linspace(-1, 1, 1000)

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(x_values, func_a(x_values))
    plt.title('Function (a)')

    plt.subplot(2, 2, 2)
    plt.plot(x_values, func_b(x_values))
    plt.title('Function (b)')

    plt.subplot(2, 2, 3)
    plt.plot(x_values, func_c(x_values))
    plt.title('Function (c)')

    plt.subplot(2, 2, 4)
    plt.plot(x_values, func_d(x_values))
    plt.title('Function (d)')

    plt.tight_layout()
    plt.show()

    def integrate_and_average(func, a, b, n):
        subinterval_width = (b - a) / n
        midpoints = np.linspace(a + subinterval_width / 2, b - subinterval_width / 2, n)
        function_values = func(midpoints)
        average_value = np.mean(function_values)

        return average_value
    
    intervals = [(0, 1), (0, 1), (-np.pi, np.pi), (-1, 1)]
    n_values = [4, 100, 200, 1000]

    for i, (a, b) in enumerate(intervals):
        print(f'\nFunction ({chr(97 + i)})')
        for n in n_values:
            average_value = integrate_and_average(locals()[f'func_{chr(97 + i)}'], a, b, n)
            print(f'n = {n}: Average value = {average_value:.4f}')

exercise1()
exercise2()
exercise3()
exercise4()
exercise5()
exercise6()
exercise7()
exercise8()