import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

def generatechen(r0, tf, dt, a, b, c):
    """Integrate a given chen system."""

    # define equations of lorenz system
    def chen(r, t):
        x = r[0]; y = r[1]; z = r[2]
        u = a * (y - x)
        v = (c - a) * x - x * z + c * y
        w = x * y - b * z 
        return np.array([u, v, w])

    ts, traj = RK4(chen, r0, tf, dt)
    return (ts, traj)

    ts, traj = RK4(chen, r0, tf, dt)
    return (ts, traj)

'''
def get_lorenz_data(tf=250, dt=0.02, skip=25, split=0.8):
    _, traj = generateLorenz((1, 1, 1), tf, dt, 10, 28, 8/3)
    
    skip_steps = int(25 / dt)
    traj = traj[skip_steps:]
    
    split_num = int(split * traj.shape[0])
    
    train_data = traj[:split_num]
    val_data = traj[split_num:]
    
    return train_data, val_data
'''
'''
def plot_lorenz(tf=250, dt=0.02, skip=25):
    _, traj = generateLorenz((1,1,1), tf, dt, 10, 20, 8/3)
    fig = figure()
    plt.plot(traj[:,1],traj[:,2],traj[:,3])
    plt.show()
if __name__=="__main__":
    plot_lorenz
'''
tf = 250
dt = 0.01
ts,f = generatechen((-0.1, 0.5, -0.6), tf, dt, 40, 3, 28)
t = np.linspace(0, 250, 25000)
x, y, z = f.T
print(x)
# Plot the Chen attractor using a Matplotlib 3D projection.
fig=plt.figure()
ax = Axes3D(fig)
ax.plot(x, y, z, 'b-', lw=0.5)
ax.set_xlabel('x', fontsize=15)
ax.set_ylabel('y', fontsize=15)
ax.set_zlabel('z', fontsize=15)
plt.tick_params(labelsize=15)
ax.set_title('chen Attractor', fontsize=15)
y_section=[]
z_section=[]
for k in range(25000):
    if abs(x[k]-0.)<4e-3:
        y_section.append(y[k])
        z_section.append(z[k])
fig = plt.figure()
plt.plot(y_section, z_section, 'ok', markersize=2)
#plt.savefig("chen.pdf")
plt.show()
