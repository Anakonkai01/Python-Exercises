import math as m 
import numpy as np


# exercise 1 :
def exer1 ():
    print("Odd Number = ",end='')
    for i in range(50,101):
        if i % 2 :
            print(f"{i}",end="  ")
        
        
# exercise 2 : 
def exer2():
    print("Number Divide by 2 and 5 =  ")
    for i in range(1500,2701):
        if i % 2  == 0 and i % 5 == 0:
            print(f"{i}",end=", ")

# exercise 3 :
def exer3():
    print("the set of number between 0 and 20 without 3 and 16 : ")
    for i in range(21):
        if i == 3 or i == 16:
            continue
        else : 
            print(i,end="  ")

# exercise 4 :
def exer4():
    countOdd = 0
    countEven = 0
    listOfNubers = []
    startN = int(input("input for First number : "))
    endN = int(input("input for Last number : "))
    for i in range(startN,endN+ 1):
        listOfNubers.append(i)
    for i in listOfNubers:
        if i % 2 :
            countOdd += 1
        else : 
            countEven += 1
    print(f"Number of odd number : {countOdd}")
    print(f"Number of even number : {countEven}")

# exercise5 :
def exer5():
    sum = 0
    for i in range(1,100):
        sum += i/(i+1)
    print(f"the result of expression is : {round(sum,2)}")

# exercise 6
def exer6():
    print("Range between 12 and 38 in numpy = ")
    a = np.arange(12,39)
    for i in a:
        print(i,end=' ')

# exercise 7
def exer7 ():
    array1 = np.array([])
    sizeArray1 = int(input("Input the Size of array 1: "))
    for i in range(sizeArray1):
        print(f"Type the value for array1[{i}] = ")
        array1 =np.append(array1,int(input()))

    array2 = np.array([])
    sizeArray2 = int(input("Input the Size of Array 2: "))
    for i in range(sizeArray2):
        print(f"Type the value for array2[{i}] = ")
        array2 =np.append(array2,int(input()))

    common_values = np.intersect1d(array1, array2).astype(int)
    print("Common values:", common_values)


def default():
    print("INVALID VALUE")

def swicthcase(choose):
    chooseExer = {
    1 : exer1,
    2 : exer2,
    3 : exer3,
    4 : exer4,
    5 : exer5,
    6 : exer6,
    7 : exer7
    }
    func = chooseExer.get(choose,default)
    func()

        
checkInput = 'y';
while (checkInput == 'y'):
    print(f"\nChoose Exercise you want execute: ");
    choose = np.arange(1,8)
    print(choose)
    chooseInput = int(input())
    swicthcase(chooseInput)
    checkInput = input(f"\nType 'y' to choose next or 'n' to stop the program: ")

