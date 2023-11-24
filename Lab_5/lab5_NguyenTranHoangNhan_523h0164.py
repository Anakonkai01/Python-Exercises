import sympy as sp
import numpy as np
import matplotlib.pyplot as plt


def exercise1():
    x = sp.symbols('x')

    # Define the functions
    fa = 4 - x**2
    fb = (x-1)**2 + 1
    fc = 1/(x**2)
    fd = (1-x)/(2*x)
    fe = (3*x)**(1/2)
    ff = (2*x + 1)**(1/2)

    # Find derivatives
    fa_prime = sp.diff(fa, x)
    fb_prime = sp.diff(fb, x)
    fc_prime = sp.diff(fc, x)
    fd_prime = sp.diff(fd, x)
    fe_prime = sp.diff(fe, x)
    ff_prime = sp.diff(ff, x)

    print(f"The derivative of ({fa}) = ({fa_prime})")
    print(f"The derivative of ({fb}) = ({fb_prime})")
    print(f"The derivative of ({fc}) = ({fc_prime})")
    print(f"The derivative of ({fd}) = ({fd_prime})")
    print(f"The derivative of ({fe}) = ({fe_prime})")
    print(f"The derivative of ({ff}) = ({ff_prime})")

def exercise2():
    x = sp.symbols('x')

    fa = x**2 + 1
    fb = x - (2*x)**2
    fc = x/(x-2)
    fd = 8/(x**2)
    fe = x**(1/2)
    ff = x**3 + 3*x
    fg = 8/((x-2)**(1/2))
    fh = 1 + (4-x)**(1/2)    

    fa_prime = sp.diff(fa,x)
    fb_prime = sp.diff(fb,x)
    fc_prime = sp.diff(fc,x)
    fd_prime = sp.diff(fd,x)
    fe_prime = sp.diff(fe,x)
    ff_prime = sp.diff(ff,x)
    fg_prime = sp.diff(fg,x)
    fh_prime = sp.diff(fh,x)

    slopefa = fa_prime.subs(x,2)
    slopefb = fb_prime.subs(x,1)
    slopefc = fc_prime.subs(x,3)
    slopefd = fd_prime.subs(x,2)
    slopefe = fe_prime.subs(x,4)
    slopeff = ff_prime.subs(x,1)
    slopefg = fg_prime.subs(x,6)
    slopefh = fh_prime.subs(x,3)

    ya_tangentline = slopefa*(x-2) + 5
    yb_tangentline = slopefb*(x-1) - 1
    yc_tangentline = slopefc*(x-3) + 3
    yd_tangentline = slopefd*(x-2) + 2
    ye_tangentline = slopefe*(x-4) + 2
    yf_tangentline = slopeff*(x-1) + 4
    yg_tangentline = slopefg*(x-6) + 4
    yh_tangentline = slopefh*(x-3) + 2

    print("tangent line of fa = ",ya_tangentline)
    print("tangent line of fb = ",yb_tangentline)
    print("tangent line of fc = ",yc_tangentline)
    print("tangent line of fd = ",yd_tangentline)
    print("tangent line of fe = ",ye_tangentline)
    print("tangent line of ff = ",yf_tangentline)
    print("tangent line of fg = ",yg_tangentline)
    print("tangent line of fh = ",yh_tangentline)

    x_vals = np.linspace(-5, 5, 400)

    plt.figure(figsize=(10, 8))
    plt.plot(x_vals, sp.lambdify(x, ya_tangentline, 'numpy')(x_vals), label=f'Tangent at fa: {sp.latex(ya_tangentline)}')
    plt.plot(x_vals, sp.lambdify(x, yb_tangentline, 'numpy')(x_vals), label=f'Tangent at fb: {sp.latex(yb_tangentline)}')
    plt.plot(x_vals, sp.lambdify(x, yc_tangentline, 'numpy')(x_vals), label=f'Tangent at fc: {sp.latex(yc_tangentline)}')
    plt.plot(x_vals, sp.lambdify(x, yd_tangentline, 'numpy')(x_vals), label=f'Tangent at fd: {sp.latex(yd_tangentline)}')
    plt.plot(x_vals, sp.lambdify(x, ye_tangentline, 'numpy')(x_vals), label=f'Tangent at fe: {sp.latex(ye_tangentline)}')
    plt.plot(x_vals, sp.lambdify(x, yf_tangentline, 'numpy')(x_vals), label=f'Tangent at ff: {sp.latex(yf_tangentline)}')
    plt.plot(x_vals, sp.lambdify(x, yg_tangentline, 'numpy')(x_vals), label=f'Tangent at fg: {sp.latex(yg_tangentline)}')
    plt.plot(x_vals, sp.lambdify(x, yh_tangentline, 'numpy')(x_vals), label=f'Tangent at fh: {sp.latex(yh_tangentline)}')

    plt.title('Tangent Lines with Labels')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()
 

    


def exercise3():
    x = sp.symbols('x')
    fa = 5*x - 3*x**2
    fb = 1/(x-1)
    fc = x**3 -2*x + 7
    fd = (x-1)/(x+1)
    
    dfa = sp.diff(fa,x)
    dfb = sp.diff(fb,x)
    dfc = sp.diff(fc,x)
    dfd = sp.diff(fd,x)

    slopefa = dfa.subs(x,1)
    slopefb = dfb.subs(x,3)
    slopefc = dfc.subs(x,-2)
    slopefd = dfc.subs(x,0)

    fax0 = fa.subs(x,1)
    fbx0 = fb.subs(x,3)
    fcx0 = fc.subs(x,-2)
    fdx0 = fd.subs(x,0)

    ya_tangentline = slopefa*(x-1) + fax0
    yb_tangentline = slopefb*(x-3) + fbx0
    yc_tangentline = slopefc*(x+2) + fcx0
    yd_tangentline = slopefd*(x) + fdx0



    print(f"slope fa = {slopefa}")
    print(f"tangent line fa = {ya_tangentline}")
    print(f"slope fb = {slopefb}")
    print(f"tangent line fb = {yb_tangentline}")
    print(f"slope fc = {slopefc}")
    print(f"tangent line fc = {yc_tangentline}")
    print(f"slope fd = {slopefd}")
    print(f"tangent line fd = {yd_tangentline}")


def exercise4():
    x = sp.symbols('x')
    h = sp.symbols('h')

    f = (-2*x**2)/3 + x
    df_definition = sp.limit((f.subs(x, 0 + h) - f.subs(x,0)) / h, h, 0)    

    print(f"the derivative of {f} at x=0 is  {df_definition}")

def exercise5():
    x = sp.symbols('x')
    h = sp.symbols('h')
    t = sp.symbols('t')
    z = sp.symbols('z')

    fx = 4 - x**2
    Fx = (x-1)**2 + 1
    gt = 1/(t**2)
    kz = (1-z)/(2*z)

    f_3_prime =  sp.limit((fx.subs(x, -3 + h) - fx.subs(x,-3)) / h, h, 0)
    f0_prime = sp.limit((fx.subs(x, 0 + h) - fx.subs(x,0)) / h, h, 0)
    f1_prime = sp.limit((fx.subs(x, 1 + h) - fx.subs(x,1)) / h, h, 0)

    F_1_prime = sp.limit((Fx.subs(x, -1 + h) - Fx.subs(x,-1)) / h, h, 0)
    F0_prime = sp.limit((Fx.subs(x, 0 + h) - Fx.subs(x,0)) / h, h, 0)
    F2_prime = sp.limit((Fx.subs(x,2 + h) - Fx.subs(x,2)) / h,h, 0)

    g_1_prime = sp.limit((gt.subs(t, -1 + h) - gt.subs(t,-1)) / h, h, 0)
    g2_prime = sp.limit((gt.subs(t, 2 + h) - gt.subs(t,2)) / h, h, 0)
    g_sqrt3_prime = sp.limit((gt.subs(t, 3**1/3 + h) - gt.subs(t,3**1/3)) / h, h, 0)

    k_1_prime = sp.limit((kz.subs(z, -1 + h) - kz.subs(z,-1)) / h, h, 0)
    k1_prime = sp.limit((kz.subs(z, 1 + h) - kz.subs(z,1)) / h, h, 0)
    k_sqrt2_prime = sp.limit((kz.subs(z, 2**1/2 + h) - kz.subs(z,2**1/2)) / h, h, 0)


    print(f"The derivative of fx at x=-3 is {f_3_prime}")
    print(f"The derivative of fx at x=0 is {f0_prime}")
    print(f"The derivative of fx at x=1 is {f1_prime}")
    print("\n")
    print(f"The derivative of Fx at x=-1 is {F_1_prime}")
    print(f"The derivative of Fx at x=0 is {F0_prime}")
    print(f"The derivative of Fx at x=2 is {F2_prime}")
    print("\n")
    print(f"The derivative of gt at t=-1 is {g_1_prime}")
    print(f"The derivative of gt at t=2 is {g2_prime}")
    print(f"The derivative of gt at t=sqrt(2) is {g_sqrt3_prime}")
    print("\n")
    print(f"The derivative of kz at z=-1 is {k_1_prime}")
    print(f"The derivative of kz at z=1 is {k1_prime}")
    print(f"The derivative of kz at z=sqrt(2) is {k_sqrt2_prime}")

def exercise6():
    x = sp.symbols('x')
    z = sp.symbols('z')

    fa = 1/(x+2)
    fb = x**2 -3*x + 4
    fc = x/(x-1)
    fd = 1 + x**1/2

    def derivative_definition(fx):
        return sp.limit((fx.subs(x,z)-fx.subs(x,x)) / (z-x), z, x)

    dfa = derivative_definition(fa)
    dfb = derivative_definition(fb)
    dfc = derivative_definition(fc)
    dfd = derivative_definition(fd)

    print(f"The derivative of fa =  {dfa}")
    print(f"The derivative of fb = {dfb}")
    print(f"The derivative of fc = {dfc}")
    print(f"The derivative of fd = {dfd}")

def exercise7():
    x = sp.symbols('x')

    fa = x**3 + 2*x
    fb = x + 5/x
    fc = x + sp.sin(2*x)
    fd = sp.cos(x) + 4*sp.sin(x)
    def execute(f,x0):
        sp.plot(f,(x,x0-1/2,x0+3))
        h = sp.symbols('h')
        q = (f.subs(x, x0 + h) - f.subs(x, x0))

        lm = sp.limit(q, h, 0)
        print("The limit of function q as h -> 0 =", lm)

        yh3 = q.subs(h, 3)*(x - x0) + f.subs(x, x0)
        yh2 = q.subs(h, 2)*(x - x0) + f.subs(x, x0)
        yh1 = q.subs(h, 1)*(x - x0) + f.subs(x, x0)

        p1 = sp.plot(f, (x, x0 - 1/2, x0 + 3), show=False, line_color='blue', label='f')
        p2 = sp.plot(yh3, show=False, line_color='green', label='yh3')
        p3 = sp.plot(yh2, show=False, line_color='red', label='yh2')
        p4 = sp.plot(yh1, show=False, line_color='purple', label='yh1')
        p1.extend(p2)
        p1.extend(p3)
        p1.extend(p4)
        p1.legend = True
        p1.show()

    execute(fa,0)
    execute(fb,1)
    execute(fc,sp.pi/2)
    execute(fd,sp.pi)


def exercise8():
    x = sp.symbols('x')
    f = x**3 -3*x +1

    #  8a 
    df = sp.diff(f,x)
    _8a_slope = df.subs(x,3)
    fdx3 = f.subs(x,3)
    _8a_tangentline = _8a_slope*(x-3) + fdx3
    print(f"8a: tangent line of fx at x=3 is {_8a_tangentline}")
    print("\n")

    #  8b
    tangentline_parallel = 9*x + 2
    dg = sp.diff(tangentline_parallel,x)
    equation = sp.Eq(dg,df)
    solutions = sp.solve(equation,x)
    for i in solutions:
        b = f.subs(x,i)
        tangentLine = 9*x + b
        print(f"8b: The tangent line of fx which parallel to 9x + 2 is:  {tangentLine}")
    print("\n")
    # 8c 
    xa = 2/3;
    ya = -1
    y_tangentline_equation = df*2/3 + f - df*x
    equation = sp.Eq(y_tangentline_equation,ya)
    solutions = sp.solve(equation,x)
    for i in solutions:
        slope = df.subs(x,i)
        y_tangentLine = slope*(x-i) + f.subs(x,i)
        print(f"8c: the Tangent line of fx which go through the point A(2/3;-1) is: {y_tangentLine}")



def exercise9():
    x = sp.symbols('x')

    f = 4*x**2 -x**3
    df = sp.diff(f,x)
    slopeAtX2 = df.subs(x,2)
    slopeAtX3 = df.subs(x,3)

    y1_tangentline = slopeAtX2*(x-2) + 8
    y2_tangentline = slopeAtX3*(x-3) + 9

    print(f"y1_tangentline = {y1_tangentline}")
    print(f"y2_tangentline = {y2_tangentline}")

    f_np = sp.lambdify(x, f, 'numpy')
    y1_tangentline_np = sp.lambdify(x, y1_tangentline, 'numpy')
    y2_tangentline_np = sp.lambdify(x, y2_tangentline, 'numpy')

    x_values = np.linspace(0, 4, 400)

    plt.plot(x_values, f_np(x_values), label='f(x) = 4x^2 - x^3')
    plt.plot(x_values, y1_tangentline_np(x_values), '--',label='Tangent at x=2')
    plt.plot(x_values, y2_tangentline_np(x_values), '--',label='Tangent at x=3')

    plt.scatter([2, 3], [f.subs(x, 2), f.subs(x, 3)], color='red', zorder=5)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Function and Tangent Lines')
    plt.legend()

    plt.show()

def exercise10():
    x = sp.symbols('x')

    f1 = (x-1)**1/3
    f2 = sp.Piecewise((-x - 2, x <= -2), (x + 2, x > -2))
    f3 = sp.Piecewise((x**2,x>= 0),(0, x<0))

    def checkDiff(f,x0):
        diff_at_x0 = sp.diff(f, x).subs(x, x0)
        if diff_at_x0 is sp.nan:
            print("The function f2(x) is not differentiable at x = ",x0)
        else:
            print("The function f2(x) is differentiable at x = ",x0)

    checkDiff(f1,0)
    checkDiff(f2,-2)
    checkDiff(f3,0)





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