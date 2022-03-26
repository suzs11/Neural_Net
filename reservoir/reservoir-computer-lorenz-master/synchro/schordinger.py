import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
h = 0.02
tau = h*h/2
NX = int(15/h)
Nt = 1000
R = np.zeros([NX+1, Nt+1])
I = np.zeros([NX+1,Nt+1])
rho = np.zeros([NX+1,Nt+1])
X = h*np.arange(NX+1)
T = tau*np.arange(Nt+1)

#initial condition
sum1=0
for i in range(NX+1):
    R[i,0] = np.exp(-1/2*((i*h-5)/0.5)**2)*np.cos(17*np.pi*i*h)
    I[i,0] = np.exp(-1/2*((i*h-5)/0.5)**2)*np.sin(17*np.pi*i*h)
    #R[i,0]=1
    #R[i,0]=0
    rho[i,0]=R[i,0]**2+I[i,0]**2
    sum1+=rho[i,0]*h

for i in range(NX+1):
    R[i,0]=R[i,0]/np.sqrt(sum1)
    I[i,0]=I[i,0]/np.sqrt(sum1)

#normalization
sum2=0
for i in range(NX+1):
    rho[i,0]=R[i,0]**2+I[i,0]**2
    sum2+=rho[i,0]*h
print(sum2)
plt.figure()
plt.plot(X,rho[:,0])
plt.title('initial')

#The boundary condition
for k in range(Nt+1):
    R[0,k] = 0
    R[NX,k]=0
    I[0,k]=0
    I[NX,k]=0
    rho[0,k]=0
    rho[NX,k]=0

for k in range(Nt):
    sum=0
    for i in range(1,NX):
        R[i,k+1]=R[i,k]-tau/h**2*(I[i-1,k]-2*I[i,k]+I[i+1,k])
        I[i,k+1]=I[i,k]-tau/h**2*(R[i-1,k]-2*R[i,k]+R[i+1,k])
        rho[i,k+1]=R[i,k+1]**2+I[i,k+1]**2
        sum+=rho[i,k+1]*h
    print(sum)
    for i in range(1,NX):
        R[i,k+1]=R[i,k+1]/np.sqrt(sum)
        I[i,k+1]=I[i,k+1]/np.sqrt(sum)
        rho[i,k+1]=rho[i,k+1]/np.sqrt(sum)

fig1 = plt.figure()
plt.contourf(rho)
plt.xlabel("T")
plt.ylabel("X")
plt.show()

def animate(k):
    line.set_ydata(rho[:,k])   #update the data
    return line
fig2=plt.figure()
line, = plt.plot(X,rho[:,0])

ani = animation.FuncAnimation(fig2,func=animate,frames=np.arange(Nt+1),interval=1)


