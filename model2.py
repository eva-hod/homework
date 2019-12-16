# coding=utf-8
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import *
import sdeint
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
K=-1
g=9.8

# Вычисления c турбулентностью и без ступеньки
a11 = -Ya
a12 = 1+Yf*K*T1/T2
a13 = Yf*K*(1-1/T2)
a14 = -Ya/L/math.sqrt(3)
a15 = -Ya/V
a21 = Mza
a22 = Mzz-Mzf*K*T1/T2
a23 = -Mzf*K*(1-1/T2)
a24 = Mza/L/math.sqrt(3)
a25 = Mza/V
a31 = 0
a32 = 1/T2
a33 = -1/T2
a34 = 0
a35 = 0
a41 = 0
a42 = 0
a43 = 0
a44 = 0
a45 = 1
a51 = 0
a52 = 0
a53 = 0
a54 = -pow(V/L,2)
a55 = -2*V/L

b11 = -Yf
b21 = Mzf
b31 = 0
b41 = 0
b51 = 0

g11 = 0
g21 = 0
g31 = 0
g41 = 0
g51 = 1
A = np.array([[a11, a12, a13, a14, a15], [a21, a22, a23, a24, a25], [a31, a32, a33, a34, a35], [a41, a42, a43, a44, a45], [a51, a52, a53, a54, a55]], float)
B = np.array([[b11], [b21], [b31], [b41], [b51]], float)
G = np.array([[g11], [g21], [g31], [g41], [g51]], float)
#t=np.linspace(0, 20)
t=[]
i=0.0
while i<21:
    t.append(i+1/16)
print t.shape()
c=0.0
Noisy = []
for i in range(50):
    Noisy.append(np.random.normal(0, 0.5))
i=0

def f(y, t):
    alfa = y[0]
    omegaz = y[1]
    e = y[2]
    u1 = y[3]
    u2 = y[4]  
    Xb=0
    f0 = a11*alfa+a12*omegaz+a13*e+a14*u1+a15*u2+b11*Xb
    f1 = a21*alfa+a22*omegaz+a23*e+a24*u1+a25*u2+b21*Xb
    f2 = a31*alfa+a32*omegaz+a33*e+a34*u1+a35*u2+b31*Xb
    f3 = a41*alfa+a42*omegaz+a43*e+a44*u1+a45*u2+b41*Xb
    f4 = a51*alfa+a52*omegaz+a53*e+a54*u1+a55*u2+b51*Xb+g51*math.sin(t)
    return [f0, f1, f2, f3, f4]

#def gg(y, t):
#    return np.array([[0], [0], [0], [0], [1]], float)
#y0=([0, 0, 0, 0, 0], float)
result=odeint(f, [0, 0, 0, 0, 0], time)

x1=np.dot(57.3,result[:,0])
x2=result[:,1]
x3=result[:,2]
#Перегрузка:
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
#Вычисления с турбулентностью со ступенькой
###############################################################################################################
def sign(x, num): 
    if x>0:
        return num
def f1(y, t):
    alfa = y[0]
    omegaz = y[1]
    e = y[2]
    u1 = y[3]
    u2 = y[4] 
    if t>2:
        Xb=sign(t, 0.1)
    else:
        Xb=0
    f0 = a11*alfa+a12*omegaz+a13*e+a14*u1+a15*u2+b11*Xb
    f1 = a21*alfa+a22*omegaz+a23*e+a24*u1+a25*u2+b21*Xb
    f2 = a31*alfa+a32*omegaz+a33*e+a34*u1+a35*u2+b31*Xb
    f3 = a41*alfa+a42*omegaz+a43*e+a44*u1+a45*u2+b41*Xb
    f4 = a51*alfa+a52*omegaz+a53*e+a54*u1+a55*u2+b51*Xb
    return [f0, f1, f2, f3, f4]

result1=odeint(f1, [0, 0, 0, 0, 0], t)

x1=np.dot(57.3, result1[:,0])
x2=np.dot(57.3, result1[:,1])
x3=np.dot(57.3, result1[:,2])
#Перегрузка:
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
