# coding=utf-8
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import *
# Константы
L=525.0
V=120.0
T1=0.4
T2=0.6
Ya=0.448
Yf=0.05
Mza=-1.877
Mzf=-1.502
Mzz=-1.155
K=-1.5
g=9.8

# Вычисления без турбулентности и без ступеньки
a11 = -Ya
a12 = 1+Yf*K*T1/T2
a13 = Yf*K*(1-1/T2)
a21 = Mza
a22 = Mzz-Mzf*K*T1/T2
a23 = -Mzf*K*(1-1/T2)
a31 = 0
a32 = 1/T2
a33 = -1/T2

b11 = -Yf
b21 = Mzf
b31 = 0

A = np.array([[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]], float)
B = np.array([[b11], [b21], [b31]], float)
t=np.linspace(0, 10)

def f(y, t):
    alfa, omegaz, e = y  
    Xb=0
    return [a11*alfa+a12*omegaz+a13*e+b11*Xb, a21*alfa+a22*omegaz+a23*e+b21*Xb, a31*alfa+a32*omegaz+a33*e+b31*Xb]

result=odeint(f, [0, 0, 0], t)

x1=np.dot(57.3,result[:,0])
x2=result[:,1]
x3=result[:,2]
ny=[]
for i in x1:
    ny.append(V*Ya/g*i)

plt.plot(t, x1)
plt.xlabel('t')
plt.ylabel('alfa')
plt.grid()
plt.show()

plt.plot(t, x2)
plt.xlabel('t')
plt.ylabel('omegaz')
plt.grid()
plt.show()

plt.plot(t, ny)
plt.xlabel('t')
plt.ylabel('ny')
plt.grid()
plt.show()
#Вычисления без турбулентности со ступенькой
###############################################################################################################
def sign(x, num): 
    if x>0:
        return num
def f1(y, t):
    alfa, omegaz, e = y  
    if t>2:
        Xb=sign(t, 0.1)
    else:
        Xb=0
    return [a11*alfa+a12*omegaz+a13*e+b11*Xb, a21*alfa+a22*omegaz+a23*e+b21*Xb, a31*alfa+a32*omegaz+a33*e+b31*Xb]

result1=odeint(f1, [0, 0, 0], t)
x1=np.dot(57.3, result1[:,0])
x2=np.dot(57.3, result1[:,1])
x3=np.dot(57.3, result1[:,2])
ny1=[]
for i in x1:
    ny1.append(V*Ya/g*i/57.3)

plt.plot(t, x1)
plt.xlabel('t')
plt.ylabel('alfa')
plt.grid()
plt.show()

plt.plot(t, x2)
plt.xlabel('t')
plt.ylabel('omegaz')
plt.grid()
plt.show()

plt.plot(t, ny1)
plt.xlabel('t')
plt.ylabel('ny')
plt.grid()
plt.show()
