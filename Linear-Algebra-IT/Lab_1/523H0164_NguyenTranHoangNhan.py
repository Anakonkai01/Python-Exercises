import sympy as sp
import numpy as np
import matplotlib.pyplot as pyplot


def exercise1():
    print("Exercise 1:")
    x = np.array([1,3,5,2,9])
    y = np.array([-1,3,15,27,29])
    print(x)
    print("The number of element in x =",len(x))
    print(y)
    print("The number of element in y =",len(y))

def exercise2():
    print()
    print("Exercise 2:")
    N = 10
    
    print("N = 10")
    a = np.arange(12,12+N*2,2)
    b = np.arange(31,31+N*2,2)
    c = np.arange(int(-N/2),int(N/2)+1,1)
    d = np.arange(int(N/2),int(-N/2)-1,-1)
    e = np.arange(N,-(N-4),-2)
    f = 1 / (2 ** np.arange(N))
    f = [sp.Rational(value) for value in f]\
    # g
    def fibonacci_sequence(n):
        fib_sequence = [1, 1]
        while len(fib_sequence) < n:
            fib_sequence.append(fib_sequence[-1] + fib_sequence[-2])
        return fib_sequence
    fibonacci_numbers = fibonacci_sequence(N)
    g = [sp.Rational(1, a) for a in fibonacci_numbers]
    
    # e
    primes = list(sp.primerange(1, 100))[:N]  
    e = [sp.Rational(1, a) for a in primes]

    # i
    i = [1]
    for index in range(2, N + 1):
        i.append(i[-1] + index)

    # j
    j = [(n ** 2) + 1 for n in range(1, N + 1)]

    j = [sp.Rational(1, i) for i in j]

    # k
    k = [sp.Rational(n, n + 1) for n in range(N)]

    # l
    l = [chr(i) for i in range(ord('a'), ord('z') + 1)]
    m = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

    print(f"a = {a}")
    print(f"b = {b}")
    print(f"c = {c}")
    print(f"d = {d}")
    print(f"e = {e}")
    print(f"f = {f}")
    print(f"g = {g}")
    print(f"e = {e}")
    print(f"i = {i}")
    print(f"j = {j}")
    print(f"k = {k}")
    print(f"l = {l}")
    print(f"m = {m}")


def exercise3():
    print()
    print()
    print("Exercise 3:")
    n = 10
    print(np.logspace(1, n, n))


def exercise4():
    print()
    print()
    print("Exercise 4:")
    x = np.array([1, 2, 3])
    y = np.array([98, 12, 33])
    z = np.concatenate((x, y), axis=0)
    print("z =",z)


def exercise5():
    print()
    print()
    print("Exercise 5:")
    x = [1, 2, 3]
    y = [4, 5, 6]
    z = np.array([x,y])
    print("z =",z)

def exercise6():
    print()
    print()
    print("Exercise 6:")
    