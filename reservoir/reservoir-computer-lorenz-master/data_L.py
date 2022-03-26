import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def generateLogistic():
    mu = np.arange(3.8, 4, 0.001)
    x, N = 0.3, 2000
    nu, traj = [], []
    for i in range(N):
        x = mu * x * (1 - x)
        if i > 1800:
            traj.append(x)
            nu.append(mu)
    traj = np.asarray(traj)
    nu = np.asarray(nu)
    return (nu, traj)

def logistic():
    mu1, N, x = 3.8, 10500, 0.5
    traj = []
    for i in range(500):
        x = mu1*x*(1 - x)
    for k in range(N-500):
        x = mu1*x*(1-x)
        traj.append(x)
    traj = np.array(traj)
    return traj



def get_logistic_data(split=0.8):
    traj = logistic()
    split_num = int(split * traj.shape[0])

    train_data = traj[:split_num]
    val_data = traj[split_num:]
    return train_data ,val_data


if __name__== "__main__":
    nu, traj = generateLogistic()
    train_data , val_data = get_logistic_data()
    print(train_data.shape)
    print(val_data.shape)
    #print(nu)
    #print(traj)
    plt.plot(nu, traj, '.k')
    plt.xlim((3.8,4.0))
    plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    plt.show()
