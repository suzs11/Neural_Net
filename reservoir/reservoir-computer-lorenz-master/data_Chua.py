import numpy as np

def RK4(f, r0, tf, dt):
    ''' Fourth-order Runge-Kutta integer
    :param f: Function to be intergrated
    :param r0: Initial condiions
    :prama tf: Integration duration
    :param da: Yimestep size
    :returns: time and trajectory vectors
    '''
    # generate an array of time steps
    ts = np.arange(0, tf, dt)
    #create an array to hold system state at each timestep
    traj = np.zeros((ts.shape[0], len(r0)))
    traj[0, :] = np.array(r0)
    # calculate system state at each time step, save it in the array
    for i in range(0, ts.shape[0] - 1):
        t = ts[i]
        r = traj[i, :]
        k1 = dt * f(r, t)
        k2 = dt * f(r + k1/2, t + dt/2)
        k3 = dt * f(r + k2/2, t + dt/2)
        k4 = dt * f(r +k3, t + dt)
        K = (1.0/6) * (k1 + 2*k2 + 2*k3 + k4)

        traj[i+1, :] = r + K
    return (ts, traj)
def generateChua(r0, tf, dt, a, b, c, d):
    '''Integrate a given Chua's system'''
    
    # define equation of lorenz system
    def chua(r, t):
        x = r[0]; y = r[1]; z = r[2]
        g = c * x + (1.0 / 2) * (d - c) * (np.abs(x + 1) - np.abs(x - 1))
        u = a * (y - x - g)
        v = x - y + z
        w = - b * y
        return np.array([u, v, w])
    ts, traj = RK4(chua, r0, tf, dt)
    return (ts, traj)
def get_chua_data(tf = 400, dt=0.2, skip=25, split=0.8):
    _, traj = generateChua((-1.6, 0, 1.6), tf, dt, 15, 25.58, -5/7, -8/7)
    skip_steps = int(25 / dt)
    traj = traj[skip_steps:]
    split_num = int(split * traj.shape[0])
    train_data = traj[:split_num]
    val_data = traj[split_num:]
    return train_data, val_data
