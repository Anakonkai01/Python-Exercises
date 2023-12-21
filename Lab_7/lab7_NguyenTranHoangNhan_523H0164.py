import sympy as sp 
import numpy as np
import matplotlib.pyplot as plt
import math
from sympy.series.sequences import SeqFormula
from sympy import AccumBounds


def exercise1():
    f_n = lambda n: 4*n + 1 
    n_array = np.arange(0, 11) 
    x_n = list( map(f_n, n_array)) 
    print(f"Exercise 1a: {x_n}")

    f_n = lambda n: n**3
    n_array = np.arange(0, 11) 
    x_n = list( map(f_n, n_array)) 
    print(f"Exercise 1b: {x_n}")
    
    f_n = lambda n: 3**n
    n_array = np.arange(0, 11) 
    x_n = list( map(f_n, n_array)) 
    print(f"Exercise 1c: {x_n}")
    
    print(f"Exercise 1d: ",end=" ")
    a = 0 
    b = 1
    while True:
        c = a + b
        a = b
        b = c
        print(f"{c}",end=", ")
        if c > 300: break

def exercise2():
    print("2a:")
    print("d = 15")
    d = 15
    print(f"a(n) = a(n-1) + d")
    a = 5
    n = 1
    while True:
        a = a + 15;  
        n += 1
        if n == 55: break
    print(f"a(55) = {a}")

    a = 5
    n = 1
    while a != 230:
        a = a + 15;  
        n += 1
        
    print(f"khi n  = {n} thi a  = 230")



    print("\n2b:")
    print("r = 2")
    print("a(n) = a(n-1)/r")
    a = 120
    n = 1
    while True:
        a = a / 2
        n += 1
        if n == 10:
            break

    x = sp.symbols('x')
    print(f"a(10) = {a}")

    
    n_solution = sp.solve(sp.Eq(15/32, 120 * (2**(1-x))), x)
    n = int(n_solution[0])  

    print(f"When n = {n}, a = 15/32")

def exercise3():

    x = sp.symbols('x')
    f = sp.cos(x)
    x0 = sp.pi/3
    order = 6
    taylor_poly = f.series(x, x0, order)
    print("Exercise3a = ",end= " ")
    print(taylor_poly)
    print()

    f = sp.ln(x)
    x0 = 2
    order = 10
    taylor_poly = f.series(x, x0, order)
    print("Exercise3b = ",end= " ")
    print(taylor_poly)
    print()

    f = sp.E**(x)
    x0 = 3
    order = 12
    taylor_poly = f.series(x, x0, order)
    print("Exercise3c = ",end= " ")
    print(taylor_poly)
    print()

def exercise4():
    x = sp.symbols('x')
    f = sp.cos(x)
    x0 = 0
    order = 6  
    Maclaurin_poly = f.series(x, x0, order)
    print("Exercise4a:",end=" ")
    print(Maclaurin_poly)
    print()

    f = sp.E**(x)
    x0 = 0
    order = 12
    Maclaurin_poly = f.series(x, x0, order)
    print("Exercise4b:",end=" ")
    print(Maclaurin_poly)
    print()

    f = 1/(1-x)
    x0 = 0
    order = 12 
    Maclaurin_poly = f.series(x, x0, order)
    print("Exercise4c:",end=" ")
    print(Maclaurin_poly)
    print()

    f = sp.atan(x)
    x0 = 0
    order = 12 
    Maclaurin_poly = f.series(x, x0, order)
    print("Exercise4d:",end=" ")
    print(Maclaurin_poly)
    print()

def exercise5():
    n = sp.symbols('n')
    fxa = (4*n**2 + 1)/(3*n**2 + 2)
    fxb = (n**2 + 1)**1/2 - n
    fxc = (2*n+ (n)**1/2)**1/2 - (2*n + 1)**1/2
    fxd = (3*5**n - 2**n)/(4**n + 2*5**n)
    fxe = (n*sp.sin(n**1/2))/(n**2 + n - 1)

    lma = sp.limit(fxa,n,sp.oo)
    lmb = sp.limit(fxa,n,sp.oo)
    lmc = sp.limit(fxb,n,sp.oo)
    lmd = sp.limit(fxd,n,sp.oo)
    lme = sp.limit(fxe,n,sp.oo)

    print(f"limit of Exercise 5a = {lma}")
    print(f"limit of Exercise 5b = {lmb}")
    print(f"limit of Exercise 5c = {lmc}")   
    print(f"limit of Exercise 5d = {lmd}")   
    print(f"limit of Exercise 5e = {lme}")        

def exercise6():
    n = sp.symbols('n')
    f1 = 1 - (0.2)**n
    f2 = (n**3)/(n**3 + 1)
    f3 = (3+5**n)/(n+n**2)
    f4 = (n**3)/(n+1)
    f5 = sp.E**(1/n)
    f6 = ((n+1)/(9*n+1))**1/2
    f7 = ((-1)**(n+1)*n)/(n+n**1/2)
    f8 = sp.tan((2*n*sp.pi)/(1+8*n))
    f9 = sp.factorial(2*n-1)/sp.factorial(2*n + 1)
    f10 = sp.ln(2*n**2 + 1) - sp.ln(n**2 + 1)
    limit_result1 = sp.limit_seq(f1, n)
    limit_result2 = sp.limit_seq(f2, n)
    limit_result3 = sp.limit_seq(f3, n)
    limit_result4 = sp.limit_seq(f4, n)
    limit_result5 = sp.limit_seq(f5, n)
    limit_result6 = sp.limit_seq(f6, n)
    limit_result7 = sp.limit_seq(f7, n)
    limit_result8 = sp.limit_seq(f8, n)
    limit_result9 = sp.limit_seq(f9, n)
    limit_result10 = sp.limit_seq(f10, n)


    
    print("Limit of the sequence 6a:", limit_result1)
    print("Limit of the sequence 6b:", limit_result2)
    print("Limit of the sequence 6c:", limit_result3)
    print("Limit of the sequence 6d:", limit_result4)
    print("Limit of the sequence 6e:", limit_result5)
    print("Limit of the sequence 6f:", limit_result6)
    print("Limit of the sequence 6g: gioi han khong xac dinh")
    print("Limit of the sequence 6h:", limit_result8)
    print("Limit of the sequence 6i:", limit_result9)
    print("Limit of the sequence 6j:", limit_result10)

def exercise7():
    a1 = 1 - (0.2)
    firstFiveTerms1 = [a1]
    for i in range(1, 5): 
        a_next = 1 - 0.2**(i + 1)
        if isinstance(a_next, complex): 
            a_next = a_next.real
        firstFiveTerms1.append(a_next)
    print(f"Exercise 7a = {firstFiveTerms1}")
    plt.plot(range(1, 6), firstFiveTerms1, marker='o', linestyle='-', color='b', label='Sequence')
    plt.title('Exercise 7a')
    plt.xlabel('n')
    plt.ylabel('a(n)')
    plt.legend()
    plt.show()


    a1 = 1
    firstFiveTerms2 = [a1]
    for i in range(1, 5): 
        a_next = (2*(i+1))/((i+1)**2 + 1)
        if isinstance(a_next, complex): 
            a_next = a_next.real
        firstFiveTerms2.append(a_next)
    print(f"Exercise 7b = {firstFiveTerms2}")
    plt.plot(range(1, 6), firstFiveTerms2, marker='o', linestyle='-', color='b', label='Sequence')
    plt.title('Exercise 7b')
    plt.xlabel('n')
    plt.ylabel('a(n)')
    plt.legend()
    plt.show()


    a1 = 1/5
    firstFiveTerms3 = [a1]
    for i in range(1, 5): 
        a_next = ((-1)**(i))/(5**(i+1))
        if isinstance(a_next, complex): 
            a_next = a_next.real
        firstFiveTerms3.append(a_next)
    print(f"Exercise 7c = {firstFiveTerms3}")
    plt.plot(range(1, 6), firstFiveTerms3, marker='o', linestyle='-', color='b', label='Sequence')
    plt.title('Exercise 7c')
    plt.xlabel('n')
    plt.ylabel('a(n)')
    plt.legend()
    plt.show()


    a1 = 1/2
    firstFiveTerms4 = [a1]
    for i in range(1, 5): 
        a_next = 1/sp.factorial(i+2)
        if isinstance(a_next, complex): 
            a_next = a_next.real
        firstFiveTerms4.append(a_next)
    print(f"Exercise 7d = {firstFiveTerms4}")
    plt.plot(range(1, 6), firstFiveTerms4, marker='o', linestyle='-', color='b', label='Sequence')
    plt.title('Exercise 7d')
    plt.xlabel('n')
    plt.ylabel('a(n)')
    plt.legend()
    plt.show()
    

    a1 = 1
    firstFiveTerms5 = [a1]
    a_n = a1
    for i in range(1, 5): 
        a_next = 5*a_n - 3
        if isinstance(a_next, complex): 
            a_next = a_next.real
        firstFiveTerms5.append(a_next)
        a_n = a_next
    print(f"Exercise 7e = {firstFiveTerms5}")
    plt.plot(range(1, 6), firstFiveTerms5, marker='o', linestyle='-', color='b', label='Sequence')
    plt.title('Exercise 7e')
    plt.xlabel('n')
    plt.ylabel('a(n)')
    plt.legend()
    plt.show()

    a1 = 2
    firstFiveTerms6 = [a1]
    a_n = a1
    for i in range(1, 5): 
        a_next = a_n/(a_n + 1)
        if isinstance(a_next, complex): 
            a_next = a_next.real
        firstFiveTerms6.append(a_next)
        a_n = a_next
    print(f"Exercise 7f = {firstFiveTerms6}")
    plt.plot(range(1, 6), firstFiveTerms6, marker='o', linestyle='-', color='b', label='Sequence')
    plt.title('Exercise 7f')
    plt.xlabel('n')
    plt.ylabel('a(n)')
    plt.legend()
    plt.show()

def exercise8():
    n = sp.symbols('n')
    a_n1 = 1 - (-2/sp.E)**n
    a_n2 = n**1/2*sp.sin(sp.pi/(n**1/2))
    a_n3 = ((3+2*n**2)/(8*n**2+n))**1/2
    a_n4 = (n**2*sp.cos(n))/(1+n**2)
    a_n5 = (2*n-1)/sp.factorial(n)
    a_n6 = (2*n-1)/((2*n)**n)

    def CheckConvergence(a_n, n):
        try:
            limit_result = sp.limit(a_n, n, sp.oo)
            if isinstance(limit_result, AccumBounds):
                return "Chua xac dinh duoc"
            elif limit_result.is_real:
                return f"Hoi Tu tai {limit_result.evalf()}"
            else:
                return "Phan Ky"
        except (NotImplementedError, TypeError):
            return "Chua xac dinh duoc"


    print(f"Exercise 8a: {CheckConvergence(a_n1, n)}")
    print(f"Exercise 8b: {CheckConvergence(a_n2, n)}")
    print(f"Exercise 8c: {CheckConvergence(a_n3, n)}")
    print(f"Exercise 8d: {CheckConvergence(a_n4, n)}")
    print(f"Exercise 8e: {CheckConvergence(a_n5, n)}")
    print(f"Exercise 8f: {CheckConvergence(a_n6, n)}")

def exercise9():
    n = sp.symbols('n')
    series = sp.Sum(4**n,(n,1,sp.oo))
    converge = series.is_convergent()
    if converge:
        print(f"Exercise 9a: Hoi tu")
    else:
        print(f"Exercise 9a: Khong Hoi tu")


    series = sp.Sum(5/(2**n),(n,1,sp.oo))
    converge = series.is_convergent()
    if converge:
        print(f"Exercise 9b: Hoi tu")
    else:
        print(f"Exercise 9b: Khong Hoi tu")

def exercise10():
    
    count = 0
    target = 21
    c = 0
    a = 0 
    b = 1
    while True:
        count += 1
        c = a + b
        a = b
        b = c
        if c == target:
            print(f"So hang tuong ung voi {target} trong day fibonnaci la so hang thu {count}")
            break
        elif c > target:
            print(f"ko tim thay trong day fibonnaci")
            break

def exercise11():
    n = sp.symbols('n')
    a_n = 28000*(1+0.03)**n
    my_list = []
    for i in range(1,4):   
        my_list.append(a_n.subs(n,i))

    for index,item in enumerate(my_list,start=1):
        print(f"Nam {index}: {item}")
    

