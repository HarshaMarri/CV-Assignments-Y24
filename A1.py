import numpy as np
import math

def mean(x):
    c=0
    for i in x:
        c=c+i
    return c/len(x)


def standard_deviation(x):
    m=mean(x)
    d=0
    for i in x:
        d=d+(i-m)**2
    D=d/len(x)
    return math.sqrt(D)

def zscore_normalisation(x):
    s=standard_deviation(x)
    m=mean(x)
    a=[]
    for i in x:
        a.append((i-m)/s)
    return a


def zscore_normalisation_2(x):
    m=np.mean(x)
    s=standard_deviation(x)
    a=[]
    for i in x:
        a.append((i-m)/s)
    return a



x = [4, 7, 7, 15, 32, 47, 63, 89, 102, 131]
print(f"Result without using numpy: {zscore_normalisation(x)}")
print(f"Result using numpy: {zscore_normalisation_2(x)}")


#####Question-2

def sigmoidfn(x):
    return 1/(1+np.exp(-x))

def derivative(x):
    s=sigmoidfn(x)
    return s*(1-s)

x=np.array([
    [9,2,5,0,0],
    [7,5,0,0,0]
])
print(f"x on applying sigmoid activation fn is: {sigmoidfn(x)}")
print(f"x on applying derivative of sigmoid activation fn is: {derivative(x)}")










