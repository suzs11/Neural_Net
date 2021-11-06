import numpy as np
import matplotlib.pyplot as plt

def couplde_chao(eps=0.26):
    N = 10000
    x1_t, x2_t, time = [], [], []
    xs, x, x2 = [], [0.6], 0.3
    for i in range(N):
        x1 = 4*x[i]*(1-x[i]) + eps*(4*x2*(1-x2)-4*x[i]*(1-x[i]))
        x.append(x1)
        #xs.append([i,x1])
        x2 = 4*x1*(1-x1) + eps*(4*x[i]*(1-x[i])-4*x1*(1-x1))
        x1_t.append(x2)
        x2_t.append(x1)
        time.append(i)
    x1_t = np.asarray(x1_t)
    x2_t = np.asarray(x2_t)
    time = np.asarray(time)
    return (time, x1_t, x2_t)

def delta():
    delta_x = []
    epsilon = np.linspace(0.7, 0.8, 50)
    for j in range(50):
        T, Y1, Y2 = couplde_chao(eps = epsilon)
        delta_sum = 0
        for i in range(len(T)):
            delta_sum += abs(Y1[i]-Y2[i])
        delta_ave = delta_sum/len(T)
    delta_x.append(delta_ave)
    delta_x = np.asarray(delta_x)
    epsilon = np.asarray(epsilon)
    return (epsilon, delta_x)


if __name__=="__main__":
    time, x1_t, x2_t = couplde_chao(eps=0.20)
    eps, delta = delta()
    plt.figure()
    plt.plot(time, x1_t,'m')
    plt.xlim([0,10000])
    plt.figure()
    plt.plot(time, x2_t,'b')
    plt.xlim([0,10000])
    plt.figure()
    plt.plot(x2_t, x1_t, '.')
    plt.figure()
    plt.plot(eps, delta.T,'r-',marker='s')
    #plt.ylim([0.0,0.4])
    plt.show()


