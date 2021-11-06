import numpy as np
import matplotlib.pyplot as plt

def global_net():
    u, N, iterate = 4, 6, 500
    ep = np.arange(0.0, 1.0, 0.01)
    epsilon = np.arange(0, 1, 0.01)
    x = np.ones((iterate,N))
    x[0,:] = np.random.random((1,N))
    print(x[0,:])
    print(sum(x[0,:]))
    delta, delta1 = np.zeros((1,iterate)), []
    tau = 1

    for eps in epsilon:
        for i in range(iterate-1):
            x[i+tau,:] = 4*(x[i,:]-x[i,:]**2)
            for j in range(N):
                for k in range(N):
                    sum1 = 0
                    if k!=j:
                        sum1 += x[i+tau,k] -x[i+tau,j]
                x1 = x[i+tau,j] +eps*(sum1/(N-1))
                x[i+tau,j] = x1
            delta[0,i] =sum((x[i+tau,:]**2)) - (sum(x[i+tau,:]))**2/N
        delta1.append(sum(delta[0,300:])/301.0)
    return (ep,delta1)

if __name__=="__main__":
    ep, delta1 = global_net()
    #print(delta1)
    #print(ep)
    plt.figure()
    plt.plot(ep, delta1,'r-',marker='s')
    plt.xlabel("$\epsilon$",fontsize=15)
    plt.ylabel("$\sigma^2$",fontsize=15)
    plt.show()
