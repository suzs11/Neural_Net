import numpy as np
import matplotlib.pyplot as plt
from logist import logist as lg
from logist import nonlinearMap as sc

def bif_logsit():
    para = np.linspace(0.0, 4.0, 100)
    results = []
    for par in para:
        result = lg(par)
        results.append(result[500:])
    #results.reshpae(500,100)
    return (para, results)

def bif_nonMap():
    mu1 = np.linspace(1.0, 5.0, 100)
    results = []
    for mu in mu1:
        result = sc(mu)
        results.append(result[50:])
    return (mu1, results)

if __name__=="__main__":
    parameters = {'axes.labelsize':20,
            'axes.titlesize':35}
    plt.rcParams.update(parameters)
    mu, x = bif_logsit()
    plt.figure()
    plt.plot(mu, x, '.k')
    plt.xlim([2.5, 4.0])
    plt.ylim([0.0,1.0])
    plt.xlabel("$\mu$")
    plt.ylabel('$x_n$')

    mu1, theta_n = bif_nonMap()
    plt.figure()
    plt.plot(mu1, theta_n, '.k')
    plt.xlim([1,5.0])
    plt.ylim([0.0,1.0])
    plt.xlabel('$\mu$')
    plt.ylabel('$\\theta_n$')
    plt.show()


