import numpy as np
import matplotlib.pyplot as plt
from pyts.image import RecurrencePlot
from scipy.spatial.distance import pdist, squareform

from data_MG import get_mg_data

##    help function      ##
def rec_plot(s, eps=1.2, steps=10):
    d = pdist(s[:,None])
    d = np.floor(d/eps)
    d[d>steps] = steps
    Z = squareform(d)
    return Z

def moving_average(s, r=5):
    return np.convolve(s, np.ones((r,))/r, mode='valid')
##########################
if __name__=="__main__":

    dt = 0.01
    train_data, val_data = get_mg_data(dt=dt)
    traj = np.r_[train_data, val_data]
    t = np.linspace(0, traj.shape[0]*dt, traj.shape[0])
    '''
    fig1 = plt.figure()
    ax1 = plt.axes(projection='3d')
    ax1.plot3D(traj[:,0],traj[:,1],traj[:,2],'r-')
    '''

    fig2 = plt.figure()
    plt.plot(traj[:-1700,0],traj[1700:,0],'m--')
    plt.xlabel("$x(t)$",fontsize=18)
    plt.ylabel("$x(t+17)$",fontsize=18)
    fig3 = plt.figure()
    plt.plot(t[:60000],traj[:60000,0], 'b-')

    '''
    x = traj[:,0]
    N=450
    eps1 = 1.
    d = 3
    t1 = 6
    rp = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            rp=np.heaviside(eps1-np.sqrt(np.sum(x[i:i+(d-1)*t1:t1]
                  - x[j:j+(d-1)*t1:t1])**2))
            rp.append()
    plt.imshow(rp, cmap='binary', origin='lower')
    plt.show()
    '''
    eps, steps = 1.0, 3
    X = traj[:10000,0]
    plt.figure(figsize=(8,8))
    plt.imshow(rec_plot(X, eps=eps, steps=steps),cmap='binary',origin='lower')
    #plt.pcolor(rec_plot(X[:1000], eps=eps, steps=steps), cmap='binary')
    plt.show()
