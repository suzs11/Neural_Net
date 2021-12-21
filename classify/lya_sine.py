import numpy as np
import matplotlib.pyplot as plt

def lya_lor():
    def sine(r):
        x=r[0]; y=r[1]
        v = y*np.sin(np.pi*x)/4.0
        w = y
        return np.array([v,w])
    r0=np.random.rand()
    Z = []; d0=1e-8
    uu = np.linspace(0,4,500)
    for u in uu:
        le=0;lsum=0
        x = np.array([r0,u])
        x1 = np.array([r0+d0,u])
        for i in range(800):
            x = sine(x)
            x1 = sine(x1)
            d1 = np.sqrt((x[0]-x1[0])**2)
            x1 = x + (d0/d1)*(x1-x)
            if i>100:
                lsum =lsum + np.log(d1/d0)
        le = lsum/(i-100)
        Z.append(le)
    return (uu,Z)

if __name__=="__main__":
    rho, lya = lya_lor()
    plt.figure()
    plt.plot(rho, lya, '.r')
    plt.xlim([0,4])
    plt.show()
