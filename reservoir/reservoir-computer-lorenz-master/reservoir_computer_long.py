import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.signal import argrelextrema
plt.rcParams['axes.unicode_minus']=False # 用来正常显示负号

#from data import get_lorenz_data
#from data_R import get_rossler_data
#from data_Chua import get_chua_data #This model needs to be trained 1000 times 
#from data_Chen import get_chen_data
from data_Liu import get_liu_data

# This file contains the ReservoirComputer class and necessary helper functions
# The main method at the bottom contains a demo of the code, run this file in the terminal to check it out

class ReservoirComputer():
    def __init__(self, dim_system=3, dim_reservoir=500, sigma=0.1, rho=1.1,density=0.05):
        self.dim_reservoir = dim_reservoir
        self.dim_system = dim_system
        self.r_state = np.zeros(dim_reservoir)
        self.A = generate_reservoir(dim_reservoir, rho, density)
        self.W_in = 2 * sigma * (np.random.rand(dim_reservoir, 3) - 0.5)
        self.W_out = np.zeros((3, dim_reservoir))

    def advance(self, u):
        """
        Generate the next r_state given an input u and the current r_state
        """

        # We decided to use the sigmoid function instead of the tanh function. It seems to provide more consisten results
        self.r_state = tanh(np.dot(self.A, self.r_state) + np.dot(self.W_in, u))

    def readout(self):
        """
        Generate and return the prediction v given the current r_state
        产生并返回给定状态的v的预测值
        """
        v = np.dot(self.W_out, self.r_state)
        return v

    def train(self, traj):
        """
        Optimize W_out so that the network can accurately model the given trajectory.
        优化W_out使得网络可以精确地给给定的轨迹建模

        Parameters
        traj: The training trajectory stored as a (n, dim_system) dimensional numpy array, where n is the number of timesteps in the trajectory
        """
        R = np.zeros((self.dim_reservoir, traj.shape[0]))
        for i in range(traj.shape[0]):
            R[:, i] = self.r_state
            x = traj[i]
            self.advance(x)
        self.W_out = lin_reg(R, traj, 0.0001)

    def predict(self, steps, actual):
        """
        Use the network to generate a series of predictions

        Parameters
        steps: the number of predictions to make. Can be any positive integer

        Returns
        predicted: the predicted trajectory stored as a (steps, dim_system) dimensional numpy array
        """
        predicted = np.zeros((steps, 3))
        for i in range(steps):
            v = self.readout()
            predicted[i] = v
            #self.advance(v)
            '''长期预测在此处修改'''
            actual_1 =np.dot(actual[i,:] ,  [[0,0,0],[0,1,0],[0,0,0]])
            actual_2 = actual_1 + np.dot(v, [[1,0,0],[0,0,0],[0,0,1]])
            self.advance(actual_2)
        return predicted


### helper functions ###

def generate_reservoir(dim_reservoir, rho, density):
    """
    Generates a random reservoir matrix with the desired rho and density

    Returns the reservoir matrix A as a (dim_reservoir, dim_reservoir) dimensional numpy array
    """
    graph = nx.gnp_random_graph(dim_reservoir, density)
    array = nx.to_numpy_array(graph) #产生一个随机网络
    rand = 2 * (np.random.rand(dim_reservoir) - 0.5)
    res = array * rand
    return scale_res(res, rho)

def scale_res(A, rho):
    """
    Scales the given reservoir matrix A such that its spectral radius is rho.
    """
    eigvalues, eigvectors = np.linalg.eig(A) #linalg.eig ：Compute the eigenvalues and right eigenvectors of a square array.
    max_eig = np.amax(eigvalues)
    max_length = np.absolute(max_eig)
    if max_length == 0:
        raise ZeroDivisionError("Max of reservoir eigenvalue lengths cannot be zero.")
    return A / max_length

def lin_reg(R, U, beta=0.0001):
    """
    Return an optimized matrix using ridge regression.

    Parameters
    R: The generated reservoir states, stored as a (3, n) dimensional numpy array
    U: The training trajectory, stored as a (n, 3) dimensional numpy array
    beta: regularization parameter
    
    Returns
    W_out: the optimized W_out array
    shape 是求出矩阵的行列数
    """
    
    Rt = np.transpose(R)  #求一个矩阵的转置
    W_out = np.dot(np.dot(np.transpose(U), Rt), np.linalg.inv(np.dot(R, Rt) + beta * np.identity(R.shape[0])))
    return W_out

def tanh(x):
    '''[xv if c else yv
      for c, xv, yv in zip(condition, x, y)]
    '''
    return (np.exp(x)-np.exp(-x)) / (np.exp(x) + np.exp(-x))


### functions for plotting results ###

def compare(predicted, actual, t, fontsize = 16):
    """
    Plot a comparison between a predicted trajectory and actual trajectory.
    
    Plots up to 9 dimensions
    """
    dimensions = predicted.shape[1]  #shape[1]是求出矩阵的列数，[0]是求出行数
    plt.clf() # Clear axis即清除当前图形中的当前活动轴。其他轴不受影响。
    plt.ion() # 打开交互模式
    
    i = 0
    while i < min(dimensions, 9):
        if i == 0:
            var = "$x$"
        elif i == 1:
            var = "$y$"
        elif i == 2:
            var = "$z$"
        else:
            var = ("dimension {}" .format((i + 1)))
        
        plt.subplot(min(dimensions, 9), 1, (i + 1))
        plt.plot(t, actual[:, i], label='actual')
        plt.plot(t, predicted[:, i],'m--', label='predict')
        #plt.axis('tight')
        plt.subplots_adjust(hspace=.001)
        plt.xlim((0,45))
        #plt.ylim((-1,1))
        plt.ylabel(var, fontsize=18)
        plt.xticks([],fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.legend(loc="upper right")
        i += 1
    plt.xticks(np.arange(0,45,step=10),fontsize=fontsize)
    plt.xlabel(r"$Time$", fontsize=18)
    plt.show()
    input("Press enter to exit")
    
def plot_poincare(predicted, actual):
    """
    Displays the poincare plot of the given predicted trajectory
    """
    plt.clf()
    plt.ion()   
    
    zp = predicted[:, 2]
    zpa = actual[:,2]

    zpmaxes = zp[argrelextrema(zp, np.greater)[0]]
    zpamaxes = zpa[argrelextrema(zpa, np.greater)[0]]

    zpi = zpmaxes[0:(zpmaxes.shape[0] - 1)]
    zpai = zpamaxes[0:(zpamaxes.shape[0] - 1)]

    zpi1 = zpmaxes[1:]
    zpai1 = zpamaxes[1:]
    
    
    plt.scatter(zpai, zpai1, label="actual")
    plt.scatter(zpi, zpi1,label="predict")
    plt.xlabel("$z_i$",fontsize=18)
    plt.ylabel("$z_{i+1}$",fontsize=18)
    plt.title("Poincare Plot")
    plt.legend(loc="upper right")
    plt.show()
    input("Press enter to exit")
    

if __name__ == "__main__":
    dt = 0.02
    #train_data, val_data = get_lorenz_data(dt=dt)
    #train_data, val_data = get_rossler_data(dt=dt)
    #train_data, val_data = get_chua_data(dt=dt)
    #train_data, val_data = get_chen_data(dt=dt)# There are some question
    train_data, val_data = get_liu_data(dt=dt)

    network = ReservoirComputer(dim_reservoir=500, density=0.25)
    network.train(train_data)
    predicted = network.predict(val_data.shape[0], val_data)
    t_grid = np.linspace(0, val_data.shape[0] * dt, val_data.shape[0])
    compare(predicted, val_data, t_grid)
    plot_poincare(predicted ,val_data)
