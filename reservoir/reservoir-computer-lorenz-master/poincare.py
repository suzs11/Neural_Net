import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import math as m
mpl.rcParams['text.usetex'] = True
fig = plt.figure()
ax = fig.gca(projection='3d')
#Surface
theta = np.linspace(0.0, 300 * np.pi, 10000)
h=1.
x = np.linspace(0.0, 2*h, 10000)
#print z
r = x*((2*h)-x)**(1./2.)
z = r * np.sin(theta)
y = r * np.cos(theta)
#Points
y2=(h**(3./2.))*np.sin(theta)
z2=(h**(3./2.))*np.cos(theta)
#ax.plot(Y, Z, X)
ax.grid(True)
plt.plot(x,y,z)
plt.plot(h+x*0,y2,z2,label=r'${\rho_4}^2+{\rho_3}^2=h^3$')
plt.plot(x*0,y*0,z*0,'.',label=r'$(0,0,0)$')
plt.plot(x*0+2*h,y*0,z*0,'.',label=r'$(2h,0,0)$')
 # Set x ticks
#plt.xticks(np.linspace(-1.2,1.2,7, endpoint=True))
ax.set_xlabel(r'$\rho_1$')
#plt.yticks(np.linspace(-1.2,1.2,7, endpoint=True))
ax.set_ylabel(r'$\rho_4$')
ax.set_zlabel(r'$\rho_3$')
plt.title(R'Critical Points of the Hamiltonian Normal Form'  )
plt.title(r'Singular Surface $\rho_3^2+\rho_4^2=\rho_1^2(2h-\rho_1)$')
#box=ax.get_position()
#ax.get_position([box.x0,box.y0,box.width,box.height])
ax.legend(loc='upper right',shadow=True, fontsize='medium')
#ax.set_zlim(-10,20)
plt.show()
