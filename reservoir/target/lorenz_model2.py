# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 19:53:16 2016

@author: bertramlee
"""

import math
import matplotlib.pyplot as plt

a=10
b=8./3.

class Position():
    def __init__(self,_r,_x0=1,_y0=0,_z0=0,_t0=0,_time=60,_dt=0.0001):
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
    def plot_zx(self,color):
        plt.plot(self.x,self.z,color)
    def plot_zy(self,color):
        plt.plot(self.y,self.z,color)
fig=plt.figure(figsize=(10,6))
ax1=plt.subplot(121)
A=Position(233)
A.calculate()
A.plot_zx('k-')
plt.title('A',fontsize=25,loc='left')
plt.xlabel(r'$x$', fontsize=18)
plt.ylabel(r'$z$', fontsize=18)
plt.legend()

ax2=plt.subplot(122)
B=Position(25)
B.calculate()
B.plot_zy('k-')
plt.title('B',fontsize=25,loc='left')
plt.xlabel(r'y',fontsize=20)
plt.ylabel(r'z',fontsize=20)
plt.legend()


plt.savefig('lorenz_2_tz.pdf')
plt.show()
