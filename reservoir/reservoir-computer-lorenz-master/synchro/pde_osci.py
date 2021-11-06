import numpy as np
import matplotlib.pyplot as plt

h, N, dt = .1, 30, 0.0001
M = 10000
A = dt/(h**2)
U =  np.zeros([N+1, M+1])
Space = np.arange(0, (N+1)*h,h)
#the boundary conditions
for k in range(0, M+1):
    U[0,k] = 0.0
    U[N,k] = 0.0

#the initial conditions
for i in range(0,N):
    U[i,0] = 4*i*h*(3 -i*h)

#recursive conditions
for k in range(0,M):
    for i in range(1,N):
        U[i, k+1]=A*U[i+1,k]+(1-2*A)*U[i,k]+A*U[i-1,k]

plt.figure()
plt.plot(Space, U[:,0],'g-')
plt.plot(Space, U[:,3000],'b-')
plt.figure()
extent = [0,1,0,3]
levels = np.arange(0,10,0.1)
plt.contourf(U, levels, origin='lower', extent=extent, cmap=plt.cm.jet)
plt.show()

