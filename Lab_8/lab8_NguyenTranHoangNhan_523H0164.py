import sympy as sp
import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def exercise1():
    x = sp.symbols('x')
    y = sp.symbols('y')
    z = sp.symbols('z')
    
    fa = x**2 + x*y**3
    fb = (x-y)/(y**2 + z**2) 
    fa.subs({x:0, y:0})
    myListFA = []
    myListFA.append(fa.subs({x:0, y:0}))
    myListFA.append(fa.subs({x:-1, y:1}))
    myListFA.append(fa.subs({x:-3, y:-2}))

    myListFB = []
    myListFB.append(fb.subs({x:3, y:-1,z:2}))
    myListFB.append(fb.subs({x:1, y:1/2,z:1/4}))
    myListFB.append(fb.subs({x:2, y:2,z:100}))

    print(f"Exercise 1a: {myListFA}\nExercise 1b: {myListFB}")

def exercise2():
    # Define the function with two variables
    def fa(x, y):
        return np.cos(x)*np.cos(y)*np.e**(((x**2 + y**2)**(1/2))/4)
    def fb(x, y):
        return -((x*y**2)/(x**2 + y**2))
    def fc(x, y):
        return (x*y*(x**2 - x**2))/(x**2 + y**2)
    def fd(x, y):
        return y**2 - y**4 - x**2
    # Generate data for plotting
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    
    Za = fa(X, Y)
    Zb = fb(X, Y)
    Zc = fc(X, Y)
    Zd = fd(X, Y)
    # Plot the function in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Za, cmap='viridis')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title('Exercise 2a')
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Zb, cmap='viridis')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title('Exercise 2b')
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Zc, cmap='viridis')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title('Exercise 2c')
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Zd, cmap='viridis')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title('Exercise 2d')
    plt.show()

def exercise3():
    x, y = sp.symbols('x y')

    # Define the symbolic expressions
    functions = [2*x**2 - 3*y - 4,
                (x**2 - 1)*(y + 2),
                x**2 - x*y + y**2,
                5*x*y - 7*x**2 - y**2 + 3*x - 6*y + 2,
                (x*y - 1)**2,
                (2*x - 3*y)**3,
                (x**2 + y**2)**1/2,
                (x**3 + y/2)**(2/3),
                1/(x+y),
                x/(x**2 + y**2),
                (x+y)/(x*y-1),
                sp.atan2(y, x),
                sp.exp(x+y+1),
                sp.exp(-x)*sp.sin(x+y),
                sp.log(x+y)]

    # Create a meshgrid for 3D plots
    x_vals = np.linspace(-5, 5, 25)
    y_vals = np.linspace(-5, 5, 25)
    X, Y = np.meshgrid(x_vals, y_vals)

    for func in functions:
        Z_func = np.array([[sp.re(func.subs({x: val_x, y: val_y})).evalf() for val_x, val_y in zip(row_x, row_y)] for row_x, row_y in zip(X, Y)])

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z_func, cmap='viridis')
        ax.set_title(str(func))
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()

    expressions_y = [2*x**2 - 3*y - 4,
                    (x**2 - 1)*(y + 2),
                    x**2 - x*y + y**2,
                    5*x*y - 7*x**2 - y**2 + 3*x - 6*y + 2,
                    (x*y - 1)**2,
                    (2*x - 3*y)**3,
                    sp.sqrt(x**2 + y**2),
                    (x**3 + y/2)**(2/3),
                    1/(x+y),
                    x/(x**2 + y**2),
                    (x+y)/(x*y-1),
                    sp.atan2(y, x),
                    sp.exp(x+y+1),
                    sp.exp(-x)*sp.sin(x+y),
                    sp.log(x+y)]

    partials_y = [sp.diff(expr_y, y) for expr_y in expressions_y]

    x_vals = np.linspace(-5, 5, 25)
    y_vals = np.linspace(-5, 5, 25)
    X, Y = np.meshgrid(x_vals, y_vals)

    for dfdy in partials_y:
        Z_dfdy = np.array([[sp.re(dfdy.subs({x: val_x, y: val_y})).evalf() for val_x, val_y in zip(row_x, row_y)] for row_x, row_y in zip(X, Y)])

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z_dfdy, cmap='viridis')
        ax.set_title(str(dfdy))
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('df/dy')
        plt.show()  

def exercise4():
    x, y = sp.symbols('x y')

    # Define the symbolic expressions and their second partial derivatives
    expressions = [x + y + x*y,
                sp.sin(x*y),
                x**2*y + sp.cos(y) + y*sp.sin(x),
                x*sp.E**y + y + 1,
                sp.ln(x+y),
                sp.atan2(y, x),
                x**2*sp.tan(x*y),
                y*sp.E**(x**2-y),
                x*sp.sin(x**2*y),
                (x-y)/(x**2 + y)]

    # Define the second partial derivatives with respect to x and y
    partials_x2 = [sp.diff(expr, x, 2) for expr in expressions]
    partials_y2 = [sp.diff(expr, y, 2) for expr in expressions]

    # Create a meshgrid for 3D plots
    x_vals = np.linspace(-5, 5, 25)
    y_vals = np.linspace(-5, 5, 25)
    X, Y = np.meshgrid(x_vals, y_vals)

    # Plot each expression, second partial derivative with respect to x, and second partial derivative with respect to y in 3D
    for expr, dfdx2, dfdy2 in zip(expressions, partials_x2, partials_y2):
        # Evaluate the expression for each point in the grid
        Z_expr = np.array([[sp.re(expr.subs({x: val_x, y: val_y})).evalf() for val_x, val_y in zip(row_x, row_y)] for row_x, row_y in zip(X, Y)])

        # Evaluate the second partial derivative with respect to x for each point in the grid
        Z_dfdx2 = np.array([[sp.re(dfdx2.subs({x: val_x, y: val_y})).evalf() for val_x, val_y in zip(row_x, row_y)] for row_x, row_y in zip(X, Y)])

        # Evaluate the second partial derivative with respect to y for each point in the grid
        Z_dfdy2 = np.array([[sp.re(dfdy2.subs({x: val_x, y: val_y})).evalf() for val_x, val_y in zip(row_x, row_y)] for row_x, row_y in zip(X, Y)])

        # Plot the expression in 3D
        fig_expr = plt.figure(figsize=(10, 8))
        ax_expr = fig_expr.add_subplot(111, projection='3d')
        ax_expr.plot_surface(X, Y, Z_expr, cmap='viridis')
        ax_expr.set_title(str(expr))
        ax_expr.set_xlabel('x')
        ax_expr.set_ylabel('y')
        ax_expr.set_zlabel('f(x, y)')
        plt.show()

        # Plot the second partial derivative with respect to x in 3D
        fig_dfdx2 = plt.figure(figsize=(10, 8))
        ax_dfdx2 = fig_dfdx2.add_subplot(111, projection='3d')
        ax_dfdx2.plot_surface(X, Y, Z_dfdx2, cmap='viridis')
        ax_dfdx2.set_title(str(dfdx2))
        ax_dfdx2.set_xlabel('x')
        ax_dfdx2.set_ylabel('y')
        ax_dfdx2.set_zlabel('d²f/dx²')
        plt.show()

        # Plot the second partial derivative with respect to y in 3D
        fig_dfdy2 = plt.figure(figsize=(10, 8))
        ax_dfdy2 = fig_dfdy2.add_subplot(111, projection='3d')
        ax_dfdy2.plot_surface(X, Y, Z_dfdy2, cmap='viridis')
        ax_dfdy2.set_title(str(dfdy2))
        ax_dfdy2.set_xlabel('x')
        ax_dfdy2.set_ylabel('y')
        ax_dfdy2.set_zlabel('d²f/dy²')
        plt.show()

def exercise5():
    x = sp.symbols('x')
    y = sp.symbols('y')

    fa = x*sp.sin(y) + y*sp.sin(x) + x*y
    fb = sp.ln(2*x + 3*y)
    fc = x*y**2 + x**2*y**3 + x**3*y**4
    fd = sp.E**x + x*sp.ln(y) + y*sp.ln(x)


    #a 
    # diff xy
    dfa1xy = sp.diff(fa,x)
    dfa2xy = sp.diff(dfa1xy,y)
    # diff yx
    dfa1yx = sp.diff(fa,y)
    dfa2yx = sp.diff(dfa1yx,x) 

    # b
    # diff xy
    dfb1xy = sp.diff(fb,x)
    dfb2xy = sp.diff(dfb1xy,y)
    # diff yx
    dfb1yx = sp.diff(fb,y)
    dfb2yx = sp.diff(dfb1yx,x) 

    # d
    # diff xy
    dfd1xy = sp.diff(fd,x)
    dfd2xy = sp.diff(dfd1xy,y)
    # diff yx
    dfd1yx = sp.diff(fd,y)
    dfd2yx = sp.diff(dfd1yx,x) 

    # c
    # diff xy
    dfc1xy = sp.diff(fc,x)
    dfc2xy = sp.diff(dfc1xy,y)
    # diff yx
    dfc1yx = sp.diff(fc,y)
    dfc2yx = sp.diff(dfc1yx,x) 

    print("Exercise 5a:",end=" ")
    if(dfa2xy == dfa2yx): print("dfaxy = dfayx")
    else: print("dfaxy != dfayx")

    print("Exercise 5b:",end=" ")
    if(dfa2xy == dfa2yx): print("dfbxy = dfbyx")
    else: print("dfbxy != dfbyx")

    print("Exercise 5c:",end=" ")
    if(dfa2xy == dfa2yx): print("dfcxy = dfcyx")
    else: print("dfcxy != dfcyx")

    print("Exercise 5d:",end=" ")
    if(dfa2xy == dfa2yx): print("dfdxy = dfdyx")
    else: print("dfdxy != dfdyx")

def exercise6():
    x = sp.symbols('x')
    y = sp.symbols('y')

    def derivative_x(f,x):
        return sp.diff(f,x,2)
    def derivative_y(f,y):
        return sp.diff(f,y,3)
    
    fa = y**2*x**4*sp.E**x + 2
    fb = y**4 + y*(sp.sin(x) - x**4)
    fc = x**5 + 5*x*y + sp.sin(x) + 7*sp.E**x
    fd = x*sp.E**(y**4/2)
    
    # a
    dfa3 = derivative_x(fa,x)
    dfa5 = derivative_y(dfa3,y)

    # b
    dfb3 = derivative_x(fb,x)
    dfb5 = derivative_y(dfb3,y)

    # c
    dfc3 = derivative_x(fc,x)
    dfc5 = derivative_y(dfc3,y)

    # d
    dfd3 = derivative_x(fd,x)
    dfd5 = derivative_y(dfd3,y)

    print(f"Exercise 6a: {dfa5}")
    print(f"Exercise 6b: {dfb5}")
    print(f"Exercise 6c: {dfc5}")
    print(f"Exercise 6d: {dfd5}")

def exercise7():
    t = sp.symbols('t')
    

    wa = sp.cos(t)**2 + sp.sin(t)**2
    wb = (sp.cos(t) + sp.sin(t))**2 + (sp.cos(t) - sp.sin(t))**2
    wc = sp.cos(t)**2/(1/t) + sp.sin(t)**2/(1/t)
    wd = 2*sp.atan(t)*sp.E**(sp.ln(t**2+1)) - sp.ln(sp.E**t)
    we = sp.E**(t-1) - sp.sin(t*sp.ln(t))  

    dwta = sp.diff(wa,t)
    dwtb = sp.diff(wb,t)
    dwtc = sp.diff(wc,t)
    dwtd = sp.diff(wd,t)
    dwte = sp.diff(we,t)

    print(f"Exercise 7a: Derivative of w(t) regard to t = : {dwta}")
    print(f"Exercise 7a: Derivative of w(t) at t = pi: {dwta.subs(t,sp.pi)}")
    print()
    print(f"Exercise 7b: Derivative of w(t) regard to t = : {dwtb}")
    print(f"Exercise 7b: Derivative of w(t) at t = 0: {dwtb.subs(t,0)}")
    print()
    print(f"Exercise 7c: Derivative of w(t) regard to t = : {dwtc}")
    print(f"Exercise 7c: Derivative of w(t) at t = 3: {dwtc.subs(t,3)}")
    print()
    print(f"Exercise 7d: Derivative of w(t) regard to t = : {dwtd}")
    print(f"Exercise 7d: Derivative of w(t) at t = 1: {dwtd.subs(t,1)}")
    print()
    print(f"Exercise 7e: Derivative of w(t) regard to t = : {dwte}")
    print(f"Exercise 7e: Derivative of w(t) at t = 1: {dwte.subs(t,1)}")

def exercise8():
    x, y, h, k = sp.symbols('x y h k') 
    f = 1 - x + y - 3*x**2*y 
    print("Exercise 8a:")
    df_dx = sp.limit((f.subs(x, x + h) - f.subs(x, x)) / h, h, 0) 
    df_dy = sp.limit((f.subs(y, y + k) - f.subs(y, y)) / k, k, 0) 
    print(f"Derivative of df/dx at (1,2): {df_dx.subs({x: 1, y: 2})}") 
    print(f"Derivative of df/dx at (1,2): {df_dy.subs({x: 1, y: 2})}") 
    print("Exercise 8b:")
    f = 4 + 2*x -3*y - 3*x*y**2 
    df_dx = sp.limit((f.subs(x, x + h) - f.subs(x, x)) / h, h, 0) 
    df_dy = sp.limit((f.subs(y, y + k) - f.subs(y, y)) / k, k, 0) 
    print(f"Derivative of df/dx at (-2,1): {df_dx.subs({x: -2, y: 1})}") 
    print(f"Derivative of df/dy at (-2,1): {df_dy.subs({x: -2, y: 1})}") 

def exercise9():
    x, y = sp.symbols('x, y') 
    f_xy = x*x - x*y + y*y/2 + 3    
    dfx = sp.diff(f_xy, x) 
    dfy = sp.diff(f_xy, y) 
    x0 = 3 
    y0 = 2 
    z0 = f_xy.subs(x, x0).subs(y, y0)      
    z = z0 + dfx.subs(x, x0).subs(y, y0)*(x-x0) + dfy.subs(x, x0).subs(y, y0)*(y-y0) 
    print("z = ", z) 
    fxy = lambda x, y: x*x - x*y + y*y/2 + 3 
    f_tangent = lambda x, y: 4*x - y - 2 
    x = np.arange(-10, 10.1, 0.1) 
    y = x.copy() 
    X,Y = np.meshgrid(x, y)  
    Z = fxy(X, Y)  
    Z_tangent = f_tangent(X, Y) 
    ax = plt.axes(projection ='3d') 
    ax.plot_surface(X, Y, Z, cmap = 'hot', linewidth=0) 
    ax.plot_surface(X, Y, Z_tangent, cmap = 'Blues', linewidth=0) 
    plt.show() 


exercise1()
exercise2()
exercise3()
exercise4()
exercise5()
exercise6()
exercise7()
exercise8()
exercise9()