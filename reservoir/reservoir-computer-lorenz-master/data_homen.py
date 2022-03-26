import numpy as np
import matplotlib.pyplot as plt

def generateHomen():
    a = np.arange(0,1.7, 0.001)
    x, y = 0.3, 0.2
    b, N  = 0.3, 500
    traj_x, traj_y, mu=[], [], []
    for i in range(N):
        x = a - x*x - b*y
        y = b*x
        if i>300:
            traj_x.append(x)
            traj_y.append(y)
            mu.append(a)
    traj_x = np.asarray(traj_x)
    traj_y = np.asarray(traj_y)
    mu = np.asarray(mu)
    return (mu, traj_x, traj_y)

if __name__=="__main__":
    mu, traj_x, traj_y = generateHomen()
    plt.figure()
    plt.plot(mu, traj_x, ',k')
    plt.plot(mu, traj_y, ',b')
    plt.xlim((1.0, 1.6))
    plt.figure()
    plt.plot(traj_x, traj_y, ",r")
    plt.show()

