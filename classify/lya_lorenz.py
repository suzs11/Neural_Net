import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import derivative


def RK4(f, r0, dt):
    # generate an array of time steps
    #ts = np.arange(0, tf, dt)
    # create an array to hold system state at each timestep
    #traj = np.zeros((ts.shape[0], len(r0)))
    #traj[0, :] = np.array(r0)
    # calculate system state at each time step, save it in the array
    #t = ts[i]
    #r = traj[i, :]
    t=0
    r=r0

    k1 = dt * f(r, t)
    k2 = dt * f(r + k1/2, t + dt/2)
    k3 = dt * f(r + k2/2, t + dt/2)
    k4 = dt * f(r + k3, t + dt)
    K = (1.0/6)*(k1 + 2*k2 + 2*k3 + k4)
    traj = r + K
    return traj

def generateLorenz_lya(r0, tf, dt, sigma,rho, beta):
    def lorenz(r, t):
        x = r[0]; y = r[1]; z = r[2]
        u = sigma * (y - x)
        v = x * (rho - z) - y
        w = x * y -beta * z
        return np.array([u, v, w])
    t = np.arange(0, tf, dt)
    traj1 = np.array((r0))
    traj1 = np.zeros((t.shape[0],len(r0)))
    traj2 = np.zeros((t.shape[0],len(r0)))
    d0=1e-11; L2=0;lya=0
    traj1[0,:]=np.array((r0)); traj2[0,:]=np.array((r0))+np.array((d0,0,0))
    for j in range(len(r0)):
        lya1=[]
        for k in range(t.shape[0]-1):
            traj1[k+1,:] = RK4(lorenz, traj1[k,:], dt)
            traj2[k+1,:] = RK4(lorenz, traj2[k,:], dt)
            d10 = (traj1[k+1,j] - traj2[k+1,j])**2
            #d11 = (traj1[k+1,1] - traj2[k+1,1])**2
            #d12 = (traj1[k+1,2] - traj2[k+1,2])**2
            d1 = np.sqrt(d10)
            diff = traj2[k+1,j] - traj1[k+1,j]
            traj2[k+1,j] = traj1[k+1,j] + (d0/d1)*diff
            if k>5000:
                L1 = np.log(abs(d1/d0))
                L2 = L1+L2
        lya1.append(L2/(k-5000))
    lya_e = max(lya1)
    return lya_e


if __name__=="__main__":
    #r0 = np.array([1,1,1])
    rho1 = np.linspace(0,250, 1000)
    lya_lor = []
    for rho in rho1:
        lya_l = generateLorenz_lya((1,1,1), 250, 0.02, 10,rho, 8.0/3)
        lya_lor.append(lya_l)
    plt.figure()
    plt.plot(rho1, lya_lor,'.r')
    plt.xlim([0,250])
    #plt.ylim([-50,10])
    plt.show()

    
