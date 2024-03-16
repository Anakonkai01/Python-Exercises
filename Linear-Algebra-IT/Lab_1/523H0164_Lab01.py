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

    x = list(range(0,21,2))

    print(f"a: {x[:6]}")
    print(f"b: {x[-5:]}")
    x_sub = [x[0],x[3],x[-1]]
    print(f"c: {x_sub}")
    x_sub = [x[0],x[2],x[4],x[6]]
    print(f"d: {x_sub}")
    print(f"e: {x[1: :2]}")
    print(f"f: {x[0: :2]}")
          

def exercise7():
    print()
    print()
    print("Exercise 7:")
    x = np.array([3,11,-9,-131,-1,1,-11,91,-6,407,-12,-11,12,153,371])
    print(f"a: {max(x)}")
    print(f"b: {min(x)}")
    print(f"c: {np.where(x > 10)[0]}")
    x_reverse = np.flipud(x)
    print(f"d: {x_reverse}")
    sorted_x_ascending = np.sort(x)
    sorted_x_descending = np.sort(x)[::-1]
    print(f"e: {sorted_x_ascending}")
    print(f"f: {sorted_x_descending}")
    count = 0
    for i in range(len(x)):
        for j in range(len(x)):
            if i != j and x[i] + x[j] == 0:
                count += 1

    # Print the count
    print(f"g: {count}")
    total_duplicates = len(x) - len(np.unique(x))
    print(f"h: {total_duplicates}")
    y = x + x[::-1]
    print(f"i: {y}")
    def is_armstrong_number(num):
        if num < 0:
            return False
        num_str = str(num)
        num_digits = len(num_str)
        sum_of_powers = sum(int(digit) ** num_digits for digit in num_str)
        return num == sum_of_powers

    w = np.array([num for num in x if is_armstrong_number(num)])
    print(f"j: {w}")
    x_nonNegative = x[x >= 0]
    print(f"k: {x_nonNegative}")
    print(f"l: {np.median(x)}")
    mean_value = np.mean(x)
    sum_below_mean = np.sum(x[x < mean_value])
    print(f"m: {sum_below_mean}")
    print(f"n: {np.abs(x)}")


exercise1()
exercise2()
exercise3()
exercise4()
exercise5()
exercise6()
exercise7()
