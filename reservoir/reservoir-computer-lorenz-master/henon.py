import numpy as np
import random
import matplotlib.pyplot as plt

def HomenMap():
    a, b, N  = 1.4, -0.3, 1500
    traj_x, traj_y=[], []
    for j in range(100000):
        random.seed(j)
        x0, y0 = random.random()*5-2.5, random.random()*5-2.5
        x , y = x0, y0
        for i in range(N):
            xn = a - x*x + b*y
            yn = x
            x, y = xn, yn
        if ((x<.295 and x>.285)or(x<1.015 and x>1.005))and\
            ((y<.295 and y>.285)or(y<1.015 and y>1.005)):
            traj_x.append(x0)
            traj_y.append(y0)
    traj_x = np.asarray(traj_x)
    traj_y = np.asarray(traj_y)
    return (traj_x, traj_y)

if __name__=="__main__":
    traj_x, traj_y = HomenMap()
    plt.figure()
    plt.plot(traj_x, traj_y, ',r')
    plt.show()



