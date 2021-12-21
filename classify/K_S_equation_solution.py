import numpy as np
L = 22
tmax = 20000; dt = 0.25; nplot=1; N=64;

x = (np.arange(0,N+1)*L/N).T
u = (np.cos(x)*(1+np.sin(x)))
v = np.fft.fft(u)

