import math as m

a = []
# convert input fraction to float
def convertFraction(num,den):
    return float(num/den)

def exe1():
    a.clear()
    print("Exercise 1.a")
    for i in range(4):
        a.append(int(input(f"Input number {i}: ")))
    print(a[0]*a[1]-a[2]+a[3])

def exe2():
    print("Exercise 1.b")
    a.clear()
    for i in range(7):
        a.append(int(input(f"Input number {i}: ")))
    sum = (a[0]+a[1])/(a[2]+a[3]) - (a[4])/(a[5]+a[6])
    print(sum)

def exe3():
    print("Exercise 1.c")
    a.clear()
    for i in range(5):
        a.append(int(input(f"Input number {i}: ")))
    cal = pow(a[0]+a[1],a[2])/(a[3]+a[4])
    print(cal)

def exe4():
    a.clear()
    print("Exercise 1.d")
    for i in range(6):
        a.append(int(input(f"Input number {i}: ")))
    cal = (pow(a[0]+a[1],a[2])-a[3])/(a[4]+a[5])



def exe5():
    a.clear()
    print("Exercise 2.a")
    for i in range(2):
        a.append(int(input("Input Number : ")))
    cal = (pow(a[0],(1/2))+pow(a[1],(1/2)))/(pow(a[0],(1/2))*pow(a[1],(1/2)))
    print(round(cal,2))


def exe6():
    print("Exercise 2.b")
    a.clear()
    for i in range(4):
        a.append(input("Input Number : "))
    num,den = a[1].split("/")
    a[1] = convertFraction(int(num),int(den))
    num,den = a[2].split("/")
    a[2] = convertFraction(int(num),int(den))
    a[0] = int(a[0])
    a[3] = int(a[3])
    cal = a[0]**a[1] +a[0]**a[2] + a[0]**a[3]
    print(cal)


PI = m.pi
E = m.e
def exe7():
    cal = m.sin(PI) + m.cos(PI)
    cal2 = cal/(m.tan(PI/4))
    print(cal2)

def exe8():
    cal = m.log(E**2,E) + m.log(1000,10) + m.log(125,5)
    print(cal)


exe1()
exe2()
exe3()
exe4()
exe5()
exe6()
exe7()
exe8()