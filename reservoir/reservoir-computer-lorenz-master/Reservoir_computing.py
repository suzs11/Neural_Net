# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 21:09:06 2020

@author: chunzhang
"""


import numpy as np
from matplotlib import pyplot as plt
from scipy.io import savemat
from scipy.integrate import odeint
import time
plt.rcParams['axes.unicode_minus']=False # 用来正常显示负号

start = time.time()    
    
n_inputs = 3
n_outputs = 3
n_reservoir = 300
spectral_radius = 1.2
sparsity = 0.98
sigma = 0.1
lam = 1e-4
alpha = 1
predicttotallen = 3200
predictforcestep = 100
predict_error = np.zeros((predicttotallen - predictforcestep, 3), dtype=np.float32)

def dmove(Point, t):
    """
    p: 位置矢量
    sets： 其他参数
    """
    x, y, z = Point
    return np.array([10.0*(y - x), x*(28.0-z)-y, x*y - 8.0/3*z])

t = np.arange(0, 500, 0.02) # 创建时间点

#在[low, high]之间使用一致分布随机取一个值
low = -10
high = 10
xinit = np.random.uniform(low, high)
yinit = np.random.uniform(low, high)
zinit = np.random.uniform(low, high)
track1 = odeint(dmove, (xinit, yinit, zinit), t)        
track2 = track1[2000:,:]
datatrain = track2[:20000]
#print('datatrain:',datatrain)
inputs_scaled = datatrain[:-1]
teachers_scaled = datatrain[1:]   
 
#------------------------W-----------------------------------------------------
# initialize recurrent weights:
# begin with a random matrix centered around zero:
W = np.random.rand(n_reservoir, n_reservoir)
# delete the fraction of connections given by (self.sparsity):
A = np.random.rand(*W.shape)
[rows, cols]=A.shape
for i in range(rows):
    for j in range(cols):
        a = A[i][j]
        if a < sparsity:
            W[i][j]=0
# compute the spectral radius of these weights:
radius = np.max(np.abs(np.linalg.eigvals(W)))
# rescale them to reach the requested spectral radius:
W = W * (spectral_radius / radius)
#print('W:',W)
#--------------------W_in------------------------------------------------------
# random input weights:
W_in = np.zeros((n_reservoir, n_inputs), dtype=np.float32)
W_in[0:n_reservoir // 3, 0] = 1
W_in[n_reservoir // 3:n_reservoir * 2 // 3, 1] = 1
W_in[n_reservoir * 2 // 3:, 2] = 1
W_in = np.multiply(W_in, np.random.rand(n_reservoir, n_inputs) * 2 * sigma - sigma)
#print('W_in:',W_in)
#-------------------W_out------------------------------------------------------    

r_states = np.zeros((inputs_scaled.shape[0], n_reservoir))
r_states_new = np.zeros((inputs_scaled.shape[0], n_reservoir))
for n in range(0, inputs_scaled.shape[0]):
    # the dynamics of the reservoir
    if n == 0:
        r_states[n, :] = (1-alpha)* np.zeros_like(r_states[0]) + alpha * np.tanh(np.dot(W, np.zeros_like(r_states[0])) + np.dot(W_in, inputs_scaled[n, :]))
    else:
        r_states[n, :] = (1-alpha)* r_states[n - 1, :] + alpha * np.tanh(np.dot(W, r_states[n - 1, :]) + np.dot(W_in, inputs_scaled[n , :]))          
r_states_new[:, ::2] = r_states[:, ::2]
r_states_new[:, 1::2] = np.square(r_states[:, 1::2])
#print('r_states:', r_states)     
transient = min(int(inputs_scaled.shape[0] / 10), 1000)
extstates_trunc = r_states_new[transient:, :]
W_out = np.dot(teachers_scaled[transient:, :].T, np.dot(extstates_trunc, np.linalg.inv(np.mat(
        np.dot(extstates_trunc.T, extstates_trunc) + np.eye(extstates_trunc.shape[1]) * lam))))

#print('W_out:', W_out) 
#------------------------predicting data---------------------------------------
t = np.arange(0, 1000, 0.02) # 创建时间点
#在[low, high]之间使用一致分布随机取一个值
low = -10
high = 10
xinit = np.random.uniform(low, high)
yinit = np.random.uniform(low, high)
zinit = np.random.uniform(low, high)
track3 = odeint(dmove, (xinit, yinit, zinit), t)       
track4 = track3[2000:,:]
datapredict_inputs = track4
#print('datapredict_inputs:', datapredict_inputs)
n_samples = datapredict_inputs[:predicttotallen].shape[0]
prediction_outputs = np.zeros((n_samples, n_outputs))
r_states_pre = np.zeros((datapredict_inputs.shape[0], n_reservoir))
sum = 0
#-------------------------predicting------------------------------------------    
for n in range(n_samples):
#--------------前forcestep步,用teacher-force来更新状态--------------------------
    if n == 0:
        r_states[n,:] = (1-alpha)* np.zeros_like(r_states[0]) + alpha * np.tanh((np.dot(W, np.zeros_like(r_states[0])) + np.dot(W_in, datapredict_inputs[n, :])))
        r_states_pre[n, ::2] = r_states[n, ::2]
        r_states_pre[n, 1::2] = np.square(r_states[n, 1::2])
        prediction_outputs[n, :] = np.dot(W_out, r_states_pre[n, :])
    elif n < predictforcestep:
        r_states[n, :] = (1-alpha)* r_states[n - 1, :] + alpha * np.tanh((np.dot(W, r_states[n - 1, :]) + np.dot(W_in, datapredict_inputs[n, :])))
        r_states_pre[n, ::2] = r_states[n,  ::2]
        r_states_pre[n, 1::2] = np.square(r_states[n, 1::2])
        prediction_outputs[n, :] = np.dot(W_out, r_states_pre[n, :])
        #end = time.time()
        #print ( 'n = %d, %f\n' % (n,end-start))
    else:
#---------从第forcestep+1步开始, 使用上一步的实际输出y(n-1)作为这一步的输入-------
        r_states[n, :] = (1-alpha) * r_states[n - 1, :] + alpha * np.tanh((np.dot(W, r_states[n - 1, :]) + np.dot(W_in, prediction_outputs[n - 1, :])))
        r_states_pre[n, ::2] = r_states[n,  ::2]
        r_states_pre[n, 1::2] = np.square(r_states[n, 1::2])
        prediction_outputs[n, :] = np.dot(W_out, r_states_pre[n, :])
        #end = time.time()
        #print ( 'n = %d, %f\n' % (n,end-start))
        sum += (prediction_outputs[n, :] - datapredict_inputs[n + 1, :])**2
predict_error = np.sqrt(sum/(predicttotallen - predictforcestep))
print("test error: \n", predict_error)


ylabellist = ["x", "y", "z"]
for figi in range(datapredict_inputs.shape[1]):
    plt.figure(figsize=(12, 8))
    xgrid = np.array(range(1, predicttotallen+1))
    plt.plot(xgrid * 0.02, datapredict_inputs[1:predicttotallen+1, figi], 'b', linewidth=2.0, linestyle='--', label="target system")
    plt.plot(xgrid * 0.02, prediction_outputs[0:, figi], 'r', linewidth=2.0, linestyle='-', label="free running ESN")
    lo, hi = plt.ylim()
    plt.legend(loc='best', fontsize=20)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.xlabel("t", fontsize=40)
    plt.ylabel(ylabellist[figi], fontsize=40)
    plt.savefig("result" + str(figi) + ".pdf")
    plt.show()

end = time.time()
print (end-start)
