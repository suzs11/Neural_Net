import numpy as np
import signalz


def mackey_glass(n, d=23, e=10, initial=0.1):
    a, b, c = 0.2, 0.1, 0.9
    x = np.zoros(n)
    x[0] = initial
    d = int(d)
    a = signal.vectorize_input(a, n)
    b = signal.vectorize_input(b, n)
    c = signal.vectorize_input(c, n)
    d = signal.vectorize_input(d, n)
    e = signal.vectorize_input(e, n)
    for k  in range(0, n-1):
        x[k+1] = c[k]*x[k] + ((a[k]*x[k-d[k]]) / (b[k]+(x[k-d[k]]**e[k])))
    return x

def RK4(f, x0, tf, dt):
    ts = np.arange(0, tf, dt)
    traj = np.zeros((ts.shape[0], 1))
    traj[0] = np.array(x0)
    tau = 17
    for i in range(0, ts.shape[0]-1):
        t = ts[i]
        x_t = traj[i]
        tau = 17
        if t <=tau:
            x_minus_tau = 0
            k1 = dt * f(x_t, x_minus_tau, t)
            k2 = dt * f(x_t + k1, x_minus_tau, t+dt/2.)
            k3 = dt * f(x_t + k2/2., x_minus_tau, t+dt/2.)
            k4 = dt * f(x_t + k3, x_minus_tau, t+dt)
            K = (1.0/6) * (k1 + 2 * k2 + 2 * k3 + k4)

            traj[i+1] = x_t + K
        else:
            x_minus_tau = traj[int((t-tau)/dt)]
            k1 = dt * f(x_t, x_minus_tau, t)
            k2 = dt * f(x_t + k1, x_minus_tau, t+dt/2.)
            k3 = dt * f(x_t + k2/2., x_minus_tau, t+dt/2.)
            k4 = dt * f(x_t + k3, x_minus_tau, t+dt)
            K = (1.0/6) * (k1 + 2 * k2 + 2 * k3 + k4)

            traj[i+1] = x_t + K

    return (ts, traj)

def generateMGlass(x0, tf, dt):
    '''the mackey glass system'''
    # the equation of mackey glass system
    def mgfun(x_t, x_minus_tau, t):
        f = - 0.1 * x_t + (0.2 * x_minus_tau) /(1 + x_minus_tau ** 10)
        return f

    ts, traj = RK4(mgfun, x0, tf, dt)
    return (ts, traj)

def get_mg_data(tf=700,dt=0.02, skip=25, split=0.8):
    _, traj = generateMGlass(1.2, tf, dt)

    skip_steps = int(skip/dt)
    traj = traj[skip_steps:]
    split_num = int(split * traj.shape[0])

    train_data = traj[:split_num]
    val_data = traj[split_num:]
    return train_data, val_data
