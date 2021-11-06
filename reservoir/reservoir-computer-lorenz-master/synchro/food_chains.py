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

def generateChain(r0, tf, dt, K,a_c,b_c,a_p,b_p,x0,y0,eps):
    """Integrate a given Lorenz system."""

    # define equations of lorenz system
    def Chain(r, t):
        x1 = r[0]; y1 = r[1]; z1 = r[2]
        x2 = r[3]; y2 = r[4]; z2 = r[5]
        u1 = x1*(1-x1/K) - (a_c*b_c*x1*y1/(x1+x0))
        v1 = a_c*y1*((b_c*x1/(x1+x0))-1)-(a_p*b_p*y1*z1/(y1+y0))+eps*(y2-y1)
        w1 = a_p*z1*(b_p*y1/(y1+y0)-1)+eps*(z2-z1)
        u2 = x2*(1-x2/K) - (a_c*b_c*x2*y2/(x2+x0))
        v2 = a_c*y2*((b_c*x2/(x2+x0))-1)-(a_p*b_p*y2*z2/(y2+y0))+eps*(y1-y2)
        w2 = a_p*z2*(b_p*y2/(y2+y0)-1)+eps*(z1-z2)
        return np.array([u1, v1, w1, u2, v2, w2])

    ts, traj = RK4(Chain, r0, tf, dt)
    return (ts, traj)

def get_chain_data(tf=250, dt=0.02, skip=25, split=0.8):
    _, traj = generateChain((1, 1, 1, 1.2,1.3,1.4), tf, dt, 10, 28, 2)
    
    skip_steps = int(25 / dt)
    traj = traj[skip_steps:]
    
    split_num = int(split * traj.shape[0])
    
    train_data = traj[:split_num]
    val_data = traj[split_num:]
    
    return train_data, val_data

def delta():
    delta_x = []
    epsilon = np.linspace(0.0, 0.012, 40)
    for j in range(40):
        T,traj=generateChain((1,1,1,1.2,1.3,1.4),tf,dt,0.99,0.4,2.009,0.08,2.876,0.16129,0.5,epsilon[j])
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
    tf, dt = 1000, 0.02
    T, traj = generateChain((1, 1, 1,1.2,1.3,1.4),tf,dt,0.99,0.4,2.009,0.08,2.876,0.16129,0.5,0.006)
    epsilon, delta_x = delta()
    plt.figure()
    plt.plot(T[500:], traj[500:,3])
    plt.xlim([200,400])
    plt.figure()
    plt.plot(T[500:], traj[500:,1])
    plt.xlim([200,400])
    plt.figure()
    plt.plot(epsilon, delta_x, 'r-',marker='s')
    plt.show()
