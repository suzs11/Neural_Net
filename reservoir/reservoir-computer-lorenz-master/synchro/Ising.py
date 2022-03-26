import numpy as np
import matplotlib.pyplot as plt

def E_Lattice(x,y):
    L = 20
    spin = np.zeros((L, L))
    E = np.zeros((L,L))
    #x, y= np.random.randint(0,L,1), np.random.randint(0,L,1)
    for i in range(L):
        for j in range(L):
            spin0 = np.random.random((L,L))
            spin[i,j] = np.where(spin0[i,j]>0.5, 1,-1)
    E = -1* spin[x,y]*(spin[(x+1)%L,y]+spin[(x-1+L)%L,]+
                      spin[x,(y+1)%L]+spin[x,(y-1+L)%L])
    return E

def E_tol():
    en = 0
    for y in range(L):
        for x in range(L):
            en += E_Lattice(x,y)
    Energy = en
    return Energy

def metroplics(steps):
    Mlist = []
    Elist = []
    L = 20
    for _ in range(steps):
        x, y = np.random.randint(0,L,1), np.random.randint(0,L,1)
        dE = -2*E_Lattice(x,y)
        if(dE<=0):
            E11 = E_Lattice(x,y)
            M = 2*E11
            E +=dE
            Mlist.append(M)
            Elist.append(E)
        elif np.random.random()<np.exp(-1.0*dE/T):
            E11 = E_Lattice(x,y)
            E11 *=-1
            M += 2*E11
            Mlist.append(M)
            Elist.append(E)

    M_list = Mlist
    E_list = Elist
    return (M_list, E_list)

def mag():
    T_list = []
    T_x = np.arange(1.0,5.0,0.01)
    for T in T_x:
        M_list, E_list = metroplics(100)
        T_list.append(np.absolute(np.average(M_list)))
    return (T_x, T_list)




if __name__=="__main__":
    T, M = mag()
    plt.figure()
    plt.plot(T,M,"-s",marker='r')
    plt.show()



