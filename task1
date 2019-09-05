from __future__ import division     
import random
import numpy as np
import matplotlib.pyplot as plt

def sign(v):
    if v >= 0:
        return 1
    else:
        return -1


def train(train_num, train_datas, lr):  
    w = [0, 0]
    b = 0
    for i in range(train_num):  
        x = random.choice(train_datas)  
        #import pdb;pdb.set_trace()
        x1, x2, y = x
        if (y * sign((w[0] * x1 + w[1] * x2 + b)) <= 0):
            w[0] += lr * y * x1
            w[1] += lr * y * x2
            # if w[1] < 0:
            #     w[1] = -1
            b += lr * y
    return w, b

def plot_points(train_datas, w, b):
    plt.figure()
    x1 = np.linspace(-4, 4, 100)
   
    x2 = (-b - w[0] * x1)/w[1]
    plt.plot(x1, x2, color='r', label='y1 data')

    datas_len = len(train_datas)
    for i in range(datas_len):
        if train_datas[i][-1] == 1:
            plt.scatter(train_datas[i][0], train_datas[i][1],marker='+',color='red', s=50)
        else:
            plt.scatter(train_datas[i][0], train_datas[i][1], marker='_',color='blue', s=50)
    plt.show()


np.random.seed(12)
num_observations = 500
x1 = np.random.multivariate_normal([0,0], [[1, .75],[.75, 1]], num_observations)
x2 = np.random.multivariate_normal([1,4], [[1, .75],[.75, 1]], num_observations)
X = np.vstack((x1, x2)).astype(np.float32)
Y = np.hstack((-np.ones(num_observations), np.ones(num_observations)))


train_datas = np.column_stack((X,Y))

 
w, b = train(train_num=10000, train_datas=train_datas, lr=0.01)
plot_points(train_datas, w, b)
