import numpy as np

def RK4(f, r0, tf, dt):
    ts = np.arange(0, tf, dt)

    traj = np.zeros((ts.shape[0], len(r0)))
    traj[0, :] = np.array(r0)

    for i in range(0, ts.shape[0] - 1):
        t = ts[i]
        r = traj[i, :]

        k1 = dt * f(r, t)
        k2 = dt * f(r + k1/2, t + dt/2)
        k3 = dt * f(r + k2/2, t + dt/2)
        k4 = dt * f(r + k3, t + dt)
        K = (1.0/6) * (k1 + 2*k2 + 2*k3 + k4)
        traj[i+1, :] = r + K
    return (ts, traj)

def generateLiu(r0, tf, dt, a , b, c):
    '''define equation of Liu system'''
    def liu(r, t):
        x =r[0]; y = r[1]; z = r[2]
        u = a * (y - x)
        v = - x * z + c * y
        w = x * y - b * z
        return np.array([u, v, w])

    ts , traj = RK4(liu, r0, tf, dt)
    return (ts, traj)


def get_liu_data(tf=250, dt=0.02, skip=25, split=0.8):
    _, traj = generateLiu((-10, 3, 37), tf, dt, 36, 3, 28.7)

    skip_steps = int(25 /dt)
    traj = traj[skip_steps:]

    split_num = int(split * traj.shape[0])

    train_data = traj[:split_num]
    val_data = traj[split_num:]
    return train_data, val_data
