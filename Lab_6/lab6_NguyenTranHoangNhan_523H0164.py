import sympy as sp 
import numpy as np 
import matplotlib.pyplot as plt



def exercise11():
    x = sp.symbols('x')
    val = sp.rad(1/x)

    f1 = sp.Piecewise(( x*sp.sin(val), x != 0) , (0, x == 0))
    f2 = sp.Piecewise(( x**2*sp.sin(val),x != 0), (0, x == 0))

    def checkDiff(f,x0):
        diff_at_x0 = sp.diff(f, x).subs(x, x0)
        if diff_at_x0 is sp.nan:
            print("The function f2(x) is not differentiable at x = ",x0)
        else:
            print("The function f2(x) is differentiable at x = ",x0)

    checkDiff(f1,0)
    checkDiff(f2,0)

def exercise12():
    x = sp.symbols('x')
    val = sp.rad(1/x)
    f = x*sp.sin(val)

    def cost(x):
        return x**3 - 6*x**2 + 15*x
    def extra(x):
        x_var = sp.symbols('x')    
        fee = cost(x_var)
        currentFee = fee.subs(x_var,x)
        newFee = fee.subs(x_var,x+1)
        extraFee = newFee - currentFee
        return extraFee

    currentPro = 10
    extraCost = extra(currentPro)
    print(f"The extra cost to produce one more radiator is: {extraCost}")



def exercise13():
    def revenue_function(x): 
        return 20000*(1-1/x)
    def marginal_revenue(x): 
        x_var = sp.symbols('x')
        
        revenue = revenue_function(x_var) 
        marginal_rev = sp.diff(revenue, x_var) 
        marginal_rev_at_x = marginal_rev.subs(x_var, x) 
        return marginal_rev_at_x

    num_machines = 100 
    marginal_rev = marginal_revenue(num_machines)
    print(f"The marginal revenue when {num_machines} machines are produced is: ${marginal_rev}")


def exercise14():
    def population_func(t): 
        return 10**6 + 10**4*t - 10**3*t**2
    def growth_rate(t): 
        t_var = sp.symbols('t')
        population = population_func(t_var)
        D_rate = sp.diff(population, t_var)
        rate = D_rate.subs(t_var, t) 
        return rate

    rate_at_0_hours = growth_rate(0) 
    print(f"The growth rate at t = 0 hours is: {rate_at_0_hours}")

exercise14()
def exercise15():

    def height_func(t): 
        return 24*t - 0.8*t**2
    
    def velocity_and_acceleration(t): 
        t_var = sp.symbols('t')
        height = height_func(t_var)
        velocity = sp.diff(height, t_var)
        acceleration = sp.diff(velocity, t_var)
        vel = velocity.subs(t_var, t)
        acc = acceleration.subs(t_var, t)
        return vel, acc
    
    def time_highest_point():
        t_var = sp.symbols('t')
        velocity = sp.diff(height_func(t_var), t_var)
        
        time = sp.solve(velocity, t_var) 
        return time[0]

    def max_height(): 
        t_max_height = time_highest_point() 
        height_max = height_func(t_max_height) 
        return height_max


    t1 = 3
    t2 = 4
    print(f"The velocity and acceleration of the rock at time t = 3 = {velocity_and_acceleration(t1)} and t = 4 = {velocity_and_acceleration(t2)}")
    print(f"the time it take the rock the to highest point is {time_highest_point()}")
    print(f"the maximum height is {max_height()}")


def exercise16():
    def newton(f, x, x0, tol, n): 
        df = sp.diff(f, x)
        for i in range(n):
            if abs(f.subs(x, x0)) < tol: 
                return x0
            x1 = x0 - f.subs(x, x0) / df.subs(x, x0) 

            if abs(x1 - x0) < tol:
                return x1
            x0 = x1
        return x0

    x = sp.symbols('x')
    f1 = 2*x**3 + 3*x - 1
    f2 = x**3 - 4
    print(float(newton(f1, x, 2, 1e-8, 1000))) 
    print(float(newton(f2, x, 2, 1e-8, 1000))) 

    f1_np = sp.lambdify(x, f1, 'numpy')
    f2_np = sp.lambdify(x, f2, 'numpy')

    x_vals = np.linspace(-2, 2, 400)

    plt.plot(x_vals, f1_np(x_vals), label='2*x**3 + 3*x - 1')
    plt.plot(x_vals, f2_np(x_vals), label='x**3 - 4')
    plt.legend()
    plt.show()

exercise16()