import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import derivative

def RK4(f, r0, tf, dt):
    """Fourth-order Runge-Kutta integrator.
    :param f: Function to be integrated
    :param r0: Initial conditions
    :param tf: Integration duration
    :param dt: Timestep size
    :returns: time and trajectory vectors
    """
    # generate an array of time steps
    ts = np.arange(0, tf, dt)
    # create an array to hold system state at each timestep
    traj = np.zeros((ts.shape[0], len(r0)))
    traj[0, :] = np.array(r0)
    # calculate system state at each time step, save it in the array
    for i in range(0, ts.shape[0]-1):
        t = ts[i]
        r = traj[i, :]

        k1 = dt * f(r, t)
        k2 = dt * f(r + k1/2, t + dt/2)
        k3 = dt * f(r + k2/2, t + dt/2)
        k4 = dt * f(r + k3, t + dt)
        K = (1.0/6)*(k1 + 2*k2 + 2*k3 + k4)

        traj[i+1, :] = r + K
    return (ts, traj)

def generateLorenz(r0, tf, dt, sigma, rho, beta):
    """Integrate a given Lorenz system."""

    # define equations of lorenz system
    def lorenz(r, t):
        x = r[0]; y = r[1]; z = r[2]
        u = sigma * (y - x)
        v = x * (rho - z) - y
        w = x * y - beta * z
        return np.array([u, v, w])

    ts, traj = RK4(lorenz, r0, tf, dt)
    return (ts, traj)

def get_lorenz_data(tf=250, dt=0.02):
    ts, traj = generateLorenz((1, 1, 1), tf, dt, 10, 70, 8.0/3)
    parameters = {'axes.labelsize': 20,
          'axes.titlesize': 35}
    plt.rcParams.update(parameters)
    plt.figure()
    plt.subplot(3,1,1)
    plt.subplots_adjust(hspace=.001)
    plt.plot(ts,traj[:,0]/traj[:,0].max())
    plt.xlim([0, 50])
    plt.ylabel('$x(t)$')
    plt.subplot(3,1,2)
    plt.plot(ts,traj[:,1]/traj[:,1].max())
    plt.ylabel('$y(t)$')
    plt.xlim([0,50])
    plt.subplot(3,1,3)
    plt.plot(ts,traj[:,2]/traj[:,2].max())
    plt.ylabel('$z(t)$')
    plt.xlim([0,50])
    plt.xlabel('$t$')
    plt.show()


if __name__=='__main__':
    get_lorenz_data()
#    rho1 = [15, 28, 160, 180]
#    for rho in rho1:
#        i = 0
#        t, traj1 = generateLorenz((1,1,1), 250, 0.02, 10, rho, 8.0/3)
#        plt.figure(i)
#        plt.plot(t,(traj1[:,0]-traj1[:,0].min())/(traj1[:,0].max()-traj1[:,0].min()))
#        plt.xlim([0, 50])
#        plt.ylim([0, 1])
#        plt.xlabel('$t$')
#        plt.ylabel('$x(t)$')
#        i+=1
#        plt.show()

    rho1 = np.linspace(0.001,250, 250)
    lya_exp = []
    for rho in rho1:
        _, traj1 = generateLorenz((1,1,1), 250, 0.02, 10, rho, 8.0/3)
        t, traj2 = generateLorenz((1+1e-8,1,1), 250, 0.02, 10, rho, 8.0/3)
        lya1 = 0; x = traj1[:,0];x_d = traj2[:,0]
        for i in range(x.shape[0]):
            lya_log = np.log(abs(x_delta[i] - x[i])/(1e-8))
            if i>2500:
                lya1 +=lya_log
        lya = lya1/(i-2500)
        lya_exp.append(lya)
    plt.figure()
    plt.plot(rho1, lya_exp,'.r')
    plt.hlines(0, 0, 250)
    plt.xlim([0,250])
    plt.xlabel('$\\rho$')
    plt.ylabel('$\lambda$')
    #plt.savefig("lya_lor.pdf")
    #plt.close()
    plt.show()
#    rho2 = np.linspace(0, 250,500)
#    X = []
#    for rho in rho2:
#        _, traj2 = generateLorenz((1,1,1), 250, 0.02, 10, rho, 8.0/3)
#        x1 = traj2[500:,0]
#        X.append(x1)
#    plt.figure()
#    plt.plot(rho2, X, ',r')
#    plt.xlim([0,250])
#    plt.xlabel("$\\rho$")
#    plt.ylabel("$\x_n$")
#    plt.savefig('bif_lorenz.pdf')
#    plt.show()
