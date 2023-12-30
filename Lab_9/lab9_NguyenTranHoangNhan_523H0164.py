import sympy as sp 
import matplotlib.pyplot as plt 
import numpy as np 


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
    fa = x**2 - 2*x - 5 # a= 0, b= 2
    fb = 3*x + x**3 + 5 # a = -4, b = 4
    fc = sp.sin(x) + 3*x**2 #a = -2, b = 2
    fd = sp.E**(x*2) + 3*x # a = -1, b =1
    fe = x**3 -3*x #a = -3, b = 0
    ff = x**3 - 3*x #a = 0, b = 3
    fg = sp.sin(x) # a = 0, b = pi
    fh = sp.sin(2*x) # a = 0, b = 2
    fi = sp.cos(x) # a = pi/2, b = 3pi/2
    fj = sp.tan(x)**2 # a = -pi/4, b = pi/4
    fk = sp.E**x*sp.sin(x) # a = 0, b = pi
    fl = x**4 - 3*x**2 # a = -4, b = 0
    fm = x**4 - 3*x**2 # a = 0, b = 4
    fn = x**5 - 5*x**3 # a = -4, b = 0
    fo = x**6 - 5*x**2 # a = -1, b = 1
    fp = x**3 - 9*x # a = -3,b b = 0
    fq = x**3 - 9*x # a = 0, b = 3
    fr = x**3 + 9*x # a = -1, b= 1


    
    
def exercise4b():
    def find_extrema_and_plot(f, a, b, title):
        # Define the variable
        x = sp.symbols('x')

        # Find the derivative
        f_prime = sp.diff(f, x)

        # Find critical points by solving f'(x) = 0
        critical_points = sp.solve(f_prime, x)

        # Find the second derivative
        f_double_prime = sp.diff(f_prime, x)

        # Check whether each critical point is a minimum or maximum
        extrema = []
        for point in critical_points:
            second_derivative_at_point = f_double_prime.subs(x, point)
            if second_derivative_at_point > 0:
                extrema.append((point, "Minimum"))
            elif second_derivative_at_point < 0:
                extrema.append((point, "Maximum"))

        print(f"\n{title} - Critical Points and Type of Extrema:")
        for point, extrema_type in extrema:
            print(f"x = {point}, {extrema_type}")

        # Evaluate the function at the critical points to find the corresponding function values
        function_values_at_critical_points = [(point, f.subs(x, point)) for point in critical_points]
        print("\nFunction Values at Critical Points:")
        for point, value in function_values_at_critical_points:
            print(f"f({point}) = {value}")

        # Generate x values for plotting
        x_values = np.linspace(a, b, 100)
        
        # Convert the sympy function to a numpy function for plotting
        f_np = sp.lambdify(x, f, 'numpy')

        # Plot the function
        plt.plot(x_values, f_np(x_values), label=f'{title}')

        # Mark the critical points
        for point, _ in extrema:
            plt.scatter(float(point), float(f.subs(x, point)), color='red', label=f'Maximum point at {point}')

        # Add labels and title
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title(f'Graph of {title}')
        plt.legend()

        # Show the plot
        plt.grid(True)
        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)
        plt.show()

    # Define the functions
    functions = [
        (x**2 - 2*x - 5, 0, 2, "fa"),
        (3*x + x**3 + 5, -4, 4, "fb"),
        (sp.sin(x) + 3*x**2, -2, 2, "fc"),
        (sp.E**(x*2) + 3*x, -1, 1, "fd"),
        (x**3 - 3*x, -3, 0, "fe"),
        (x**3 - 3*x, 0, 3, "ff"),
        (sp.sin(x), 0, sp.pi, "fg"),
        (sp.sin(2*x), 0, 2, "fh"),
        (sp.cos(x), sp.pi/2, 3*sp.pi/2, "fi"),
        (sp.tan(x)**2, -sp.pi/4, sp.pi/4, "fj"),
        (sp.E**x*sp.sin(x), 0, sp.pi, "fk"),
        (x**4 - 3*x**2, -4, 0, "fl"),
        (x**4 - 3*x**2, 0, 4, "fm"),
        (x**5 - 5*x**3, -4, 0, "fn"),
        (x**6 - 5*x**2, -1, 1, "fo"),
        (x**3 - 9*x, -3, 0, "fp"),
        (x**3 - 9*x, 0, 3, "fq"),
        (x**3 + 9*x, -1, 1, "fr"),
    ]

    for func, a, b, title in functions:
        find_extrema_and_plot(func, a, b, title)