# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 19:45:23 2016
@author: bertramlee
"""

import matplotlib.pyplot as plt

a=10
b=8./3.

class Position():
    def __init__(self,_r,_x0=1,_y0=0,_z0=0,_t0=0,_time=500,_dt=0.0001):
        self.x=[_x0]
        self.y=[_y0]
        self.z=[_z0]
        self.t=[_t0]
        self.r=_r
        self.time=_time
        self.dt=_dt
        self.n=int(self.time/self.dt)
    def calculate(self):
        for i in range(self.n):
            self.x.append(self.x[-1]+a*(self.y[-1]-self.x[-1])*self.dt)
            self.y.append(self.y[-1]+(-self.x[-2]*self.z[-1]+self.r*self.x[-2]-self.y[-1])*self.dt)
            self.z.append(self.z[-1]+(self.x[-2]*self.y[-2]-b*self.z[-1])*self.dt)
            self.t.append(self.t[-1]+self.dt)
    def plot_zy(self):
        y_section=[]
        z_section=[]
        for i in range(len(self.t)):
            if abs(self.x[i]-0.)<4E-3:
                y_section.append(self.y[i])
                z_section.append(self.z[i])
        plt.plot(y_section,z_section,'ok',markersize=2)
    def plot_zx(self):
        x_section=[]
        z_section=[]
        for i in range(len(self.t)):
            if abs(self.y[i]-0.)<4E-3:
                x_section.append(self.x[i])
                z_section.append(self.z[i])
        plt.plot(x_section,z_section,'ok',markersize=2)

plt.figure(figsize=(8,8))
a1=plt.subplot(211)
A=Position(25)
A.calculate()
A.plot_zy()
plt.title('Phase space plot:z vetsus y when x=0')
plt.xlabel('y')
plt.ylabel('z')
plt.ylim(0,30)
plt.xlim(-10,10)
plt.legend()

a2=plt.subplot(212)
B=Position(25)
B.calculate()
B.plot_zx()
plt.title('Phase space plot:z vetsus x when y=0')
plt.xlabel('x')
plt.ylabel('z')
plt.ylim(0,40)
plt.xlim(-20,20)
plt.legend()

plt.savefig('lorenz3_tz.pdf')
plt.show()
