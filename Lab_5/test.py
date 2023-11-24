import sympy as sp

# Define the symbol
x = sp.symbols('x')

# Define a piecewise function
fa = (x-1)**1/3 
fb = sp.Piecewise((-(x+2), x <= -2), ((x+2), x > -2))
fc = sp.Piecewise((x**2, x >= 0), (0,x < 0))


dfa = sp.diff(fa, x)
dfb = sp.diff(fb, x)
dfc = sp.diff(fc, x)

derivative_at_point = dfa.subs(x, 1)

if sp.limit(dfa, x, 1, dir='+') == sp.limit(dfa, x, 1, dir='-'):
    print(f"The function is differentiable at x = 1. The result of derivative fx at x=1 is : {derivative_at_point}")
else:
    print("The function is not differentiable at x = 1.")


for i in range(-10,0.1,0.5): 
    if sp.limit(dfb, x, -2, dir='+') == sp.limit(dfa, x, -2, dir='-'):
        print(f"The function is differentiable at x = 1. The result of derivative fx at x=-2 is : {derivative_at_point}")
    else:
        print("The function is not differentiable at x = 1.")

# if sp.limit(dfb, x, 0, dir='+') == sp.limit(dfa, x, 0, dir='-'):
#     print(f"The function is differentiable at x = 0. The result of derivative fx at x=0 is : {derivative_at_point}")
# else:
#     print("The function is not differentiable at x = 0.")


