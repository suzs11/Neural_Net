# -*- conding=utf-8 -*-
'''
Time:2020.10.27
author:ZS
Project:Numerical Integration of Lorentz ODE
'''

import numpy as np
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy import io
import scipy.io as sio

def lorentz(Point, t, sets):
    """
    point：present location index
    sets：super parameters
    """
    p, r, b = sets
    x, y, z = Point
    return np.array([p * (y - x), x * (r - z), x * y - b * z])


t = np.arange(0, 60, 0.01)
P1 = odeint(lorentz, (0., 1., 0.), t, args=([10., 28., 3.],))  #
## (0.,1.,0.) is the initial point; args is the set for super parameters
P2 = odeint(lorentz, (0., 1.01, 0.), t, args=([10., 28., 3.],))


plt.subplot(3,1,1)
plt.plot(t,P1[:,0])

plt.subplot(3,1,2)
plt.plot(t,P1[:,1])

plt.subplot(3,1,3)
plt.plot(t,P1[:,2])

#plt.savefig('lorentz.pdf')
plt.show()

#a=P1[:,0]
#b=P1[:,1]
#c=P1[:,2]
#sio.savemat('lor.mat',{'a':a,'b':b,'c':c})

np.savetxt('lor',P1,delimiter=',')
