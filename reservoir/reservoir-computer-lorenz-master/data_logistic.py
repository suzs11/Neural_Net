import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import matplotlib.ticker as ticker

def LogisticMap(mu_1, mu_2):
    mu = np.arange(mu_1, mu_2, 0.001)
    #print(mu)
    iters = 1000 # 不进行输出的迭代次数
    last = 100 # 最后画出结果的迭代次数
    traj = np.zeros((mu.shape[0], iters+last))
    for j in range(0, mu.shape[0]):
        x = 0.6
        for i in tqdm(range(0,iters+last)):
            x = mu[j] * x * (1 - x)
            traj[j, i] = x
    return (mu,traj)

def get_logistic_data():
    mu, traj = LogisticMap(3.8, 4)
    #skip_steps = int(25 / dt)
    skip_steps = 200
    traj = traj[:,1000:]
    split_num = int(0.8 * traj.shape[0])

    train_data = traj[:split_num,:]
    val_data = traj[split_num:,:]
    return (train_data, val_data)

if __name__=="__main__":
    train_data, val_data = get_logistic_data()
    print(train_data.shape)
    print(val_data.shape)
    mu, traj = LogisticMap(3.8, 4)

    plt.figure()
    plt.plot(mu, traj[:,1000:],'o',markersize=3., color='k')
    font1 = {'family' : 'serif',
    'weight' : 'normal',
    'size'   : 20,
    }
    plt.xlim((3.8, 4.0))
    plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('$\mu$', font1)
    plt.ylabel('$x$',font1)
    #plt.savefig('l.pdf')
    plt.show()
