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
    fa = x**2 - 2*x - 5
    fb = 3*x + x**3 + 5
    fc = sp.sin(x) + 3*x**2 
    fd = sp.E**(x*2) + 3*x
    fe = x**3 -3*x
    ff = x**3 - 3*x
    fg = sp.sin(x)
    fh = sp.sin(2*x)
    fi = sp.cos(x) 
    fj = sp.tan(x)**2
    fk = sp.E**x*sp.sin(x)
    fl = x**4 - 3*x**2
    fm = x**4 - 3*x**2
    fn = x**5 - 5*x**3
    fo = x**6 - 5*x**2
    fp = x**3 - 9*x
    fq = x**3 - 9*x
    fr = x**3 + 9*x
