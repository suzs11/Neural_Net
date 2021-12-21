
import numpy as np
 
data=['a','b','c','a','a','b']
data1=np.array(data)#将list转换为array，使用属性shape，用于统计元素个数
 
#计算信息熵的方法
def calc_ent(x):
    """
        calculate shanno ent of x
    """
    #x.shape[0]计算数组x的元素长度，x长度为x.shape[0]=6
    #set() 函数创建一个无序不重复元素集
    #得到数组x的元素（不包含重复元素），即x_value_list={'c', 'b', 'a'}
    x_value_list = set([x[i] for i in range(x.shape[0])])
    #print(x_value_list)
    ent = 0.0
    for x_value in x_value_list:
        p = float(x[x == x_value].shape[0]) / x.shape[0]#计算每个元素出现的概率
    #print(p)
        logp = np.log2(p)
        ent -= p * logp
    print(ent)
 
if __name__ == '__main__':
    calc_ent(data1)

