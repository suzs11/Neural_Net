import numpy as np
import matplotlib.pyplot as plt
import math

def logist(mu):
    N = 1500
    x = np.zeros(N+1)
    x[0] = 0.5
    for i in range(N):
        x[i+1] = mu * x[i]*(1 - x[i])
    return x
def lya_logist(mu_l):
    N = 1500
    x = 0.5; sum1 = 0
    for j in range(N):
        x = mu_l*x*(1 - x)
        sum1 = sum1 + np.log(abs(mu_l*(1 - 2 * x)))
    sum2 = sum1/N
    return sum2
def shannon_logist(x):
    x_value_list = set([x[i] for i in range(x.shape[0])])
    ent = 0.0
    for x_value in x_value_list:
        p = float(x[x==x_value].shape[0]) / x.shape[0]
        logp = np.log2(p)
        ent -= p*logp/np.log2(x.shape[0])
    return ent
#ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
#\theta_{n+1} = \theta_n + \Omega - mu/2*pi*sin(2*pi*\theta_n)
#\theta_{n+1} = \theta_{n+1} % 1.0
#ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

def nonlinearMap(mu1):
    ome = 0.606661
    N =1500
    #theta = np.zeros(N+1)
    #I = np.zeros(N+1)
    #theta[0] = 0.6
    #I[0] = 0
    theta = 0.6; I = 0
    Is = []; thetas = []
    for i in range(N):
        Is.append(I)
        thetas.append(theta)
        theta =theta + ome - (mu1*np.sin(2*np.pi*theta)/(2*np.pi))
        #theta = theta + I
        #I = I % 1.0
        theta = theta % 1.0
    return thetas

if __name__=="__main__":
    x_n = logist(3.5)
    parameters = {'axes.labelsize': 20,
          'axes.titlesize': 35}
    plt.rcParams.update(parameters)
    plt.figure()
    plt.plot(x_n)
    plt.xlim([0,200])
    plt.ylim([0,1])
    plt.xlabel('$n$')
    plt.ylabel('$X_n$')

    x_n1 = logist(3.8)
    plt.figure()
    plt.plot(x_n1)
    plt.xlim([0,200])
    plt.ylim([0,1])
    plt.xlabel('$n$')
    plt.ylabel('$X_n$')

    mu_l = np.linspace(0, 4.0, 1000)
    lamb = []
    for mu_l1 in mu_l:
        lam = lya_logist(mu_l1)
        lamb.append(lam)
    plt.figure()
    plt.plot(mu_l, lamb, '.r')
    plt.xlabel("$\mu$")
    plt.ylabel("$\lambda$")
    plt.hlines(0.0, 2.5, 4.0)
    plt.xlim([2.5,4.0])

    shannon_ent = []
    for mu_l2 in mu_l:
        data = logist(mu_l2)
        ent = shannon_logist(data)
        shannon_ent.append(ent)
    plt.figure(figsize = (8, 4.9))
    plt.plot(mu_l, shannon_ent,'.b')
    plt.xlabel('$\mu$')
    plt.ylabel('Entropy')
    plt.xlim([2.5, 4.0])
    plt.hlines(0.75, 2.5,4.0)


    thetas = nonlinearMap(2.333)
    plt.figure()
    plt.plot(thetas)
    plt.xlim([0,200])
    plt.ylim([0,1])
    plt.xlabel('$n$')
    plt.ylabel('$\\theta_n$')

    plt.show()
