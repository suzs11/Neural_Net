import numpy as np
import matplotlib.pyplot as plt

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

def generateLorenz(r0, tf, dt, sigma, rho, beta, eps):
    """Integrate a given Lorenz system."""

    # define equations of lorenz system
    def lorenz(r, t):
        x1 = r[0]; y1 = r[1]; z1 = r[2]
        x2 = r[3]; y2 = r[4]; z2 = r[5]
        u1 = sigma * (y1 - x1) + eps*(x2-x1)
        v1 = x1 * (rho - z1) - y1 + eps*(y2-y1)
        w1 = x1 * y1 - beta * z1 + eps*(z2-z1)
        u2 = sigma * (y2 - x2) + eps*(x1-x2)
        v2 = x2 * (rho - z2) - y2 + eps*(y1-y2)
        w2 = x2 * y2 - beta * z2 + eps*(z1-z2)
        return np.array([u1, v1, w1, u2, v2, w2])

    ts, traj = RK4(lorenz, r0, tf, dt)
    return (ts, traj)

def get_lorenz_data(tf=250, dt=0.02, skip=25, split=0.8):
    _, traj = generateLorenz((1, 1, 1, 1.2,1.3,1.4), tf, dt, 10, 28, 2)
    
    skip_steps = int(25 / dt)
    traj = traj[skip_steps:]
    
    split_num = int(split * traj.shape[0])
    
    train_data = traj[:split_num]
    val_data = traj[split_num:]
    
    return train_data, val_data

def delta():
    delta_x = []
    epsilon = np.linspace(0.2, 0.50, 30)
    for j in range(30):
        T, traj = generateLorenz((1,1,1,1.2,1.3,1.4),tf,dt,10,28,2,epsilon[j])
        delta_sum = 0
        Y1 = traj[500:,0]
        Y2 = traj[500:,3]
        T = T[500:]
        for i in range(len(T)):
            delta_sum += abs(Y1[i]-Y2[i])
        delta_ave = delta_sum/len(T)
        delta_x.append(delta_ave)
    delta_x = np.asarray(delta_x)
    epsilon = np.asarray(epsilon)
    return (epsilon, delta_x)

if __name__=="__main__":
    tf, dt = 5000, 0.02
    T, traj = generateLorenz((1, 1, 1,1.2,1.3,1.4),tf, dt, 10, 28, 2, 0.30)
    epsilon, delta_x = delta()
    plt.figure()
    plt.plot(T[2500:], traj[2500:,3])
    plt.xlim([200,400])
    plt.figure()
    plt.plot(T[2500:], traj[2500:,1])
    plt.xlim([200,400])
    plt.figure()
    plt.plot(epsilon, delta_x, 'r-',marker='s')
    plt.show()
