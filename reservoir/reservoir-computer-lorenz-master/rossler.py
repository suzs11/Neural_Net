# -*- conding=utf-8 -*-
'''
Time:2020.10.27
author:ZS
Project:Numerical Integration for R\"osser ODE
'''
import numpy as np
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def rossler(Point, t, sets):
    """
    point：present location index
    sets：super parameters
    """
    a, b, c = sets
    x, y, z = Point
    return np.array([- y - z, x + a*y, b + z * (x - c)])


t = np.arange(0, 720, 0.1)
P1 = odeint(rossler, (0., 1., 0.), t, args=([0.2, 0.2, 9.],))  #
## (0.,1.,0.) is the initial point; args is the set for super parameters
P2 = odeint(rossler, (0., 1.01, 0.), t, args=([0.2, 0.2, 9.0],))


plt.figure()

plt.subplot(3,1,1)
plt.plot(t,P1[:,0])
plt.xlim(0,300)
plt.subplot(3,1,2)
plt.plot(t,P1[:,1])
plt.xlim(0,300)

plt.subplot(3,1,3)
plt.plot(t,P1[:,2])
plt.xlim(0,300)

plt.savefig('rossler.pdf')
plt.show()

np.savetxt('ros',P1,delimiter=',')
