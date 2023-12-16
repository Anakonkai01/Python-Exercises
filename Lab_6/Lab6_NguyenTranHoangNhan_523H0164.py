import sympy as sp
import numpy as np 
import matplotlib.pyplot as plt

def exercise11():
    x = sp.symbols('x')

    f1 = sp.Piecewise((x*sp.sin(1/(sp.rad(x))), x != 0), (0, x  == 0))
    f2 = sp.Piecewise((x**2*sp.sin(1/(sp.rad(x))),x != 0),(0, x == 0))

    def checkDiff(f,x0):
        diff_at_x0 = sp.diff(f, x).subs(x, x0)
        if diff_at_x0 is sp.nan:
            print(f"The function {f} is not differentiable at x = ",x0)
        else:
            print(f"The function {f} is differentiable at x = ",x0)

    checkDiff(f1,0)
    checkDiff(f2,0)
    
def exercise12():
    