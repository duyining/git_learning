'''
@Author: 一蓑烟雨任平生
@Date: 2020-01-02 16:02:57
@LastEditTime : 2020-01-06 18:27:12
@LastEditors  : Please set LastEditors
@Description: In User Settings Edit
@FilePath: /machine_learning_by_hand/perceptron.py
'''

import numpy as np
import math
from sklearn.datasets import load_iris

def loaddata():
    dataArr = []
    labelArr = []
    iris = load_iris()
    label = iris['target']
    for item in label:
        if item == 0:
            labelArr.append(1)
        elif item == 1 :
            labelArr.append(-1)
    dataArr = iris.data[0:100]
    return dataArr, labelArr

def train_perceptron(dataArr, labelArr, Maxiter=400, toler=0.001, eta = 0.4):
    dataMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).T
    m,n = np.shape(dataMat)
    W = np.mat(np.zeros((1, n)))
    b = -1 #给b初值定义为-1 保证算法开始能够更新
    iter = 0
    print(np.shape(dataMat))
    while iter < Maxiter:
        for i in range(m):
            if labelMat[i,:].tolist()[0][0] * (np.dot(W, dataMat[i,:].T).tolist()[0][0] + b) <= 0:
                W += eta * labelMat[i,:].tolist()[0][0] * dataMat[i,:]
                b += eta * labelMat[i,:].tolist()[0][0]
                break
        iter += 1 
    return W, b 
if __name__ == "__main__":
    dataArr, labelArr = loaddata()
    W, b = train_perceptron(dataArr, labelArr)
    for i in range(100):
        print(labelArr[i] * (np.dot(W, dataArr[i,:]).tolist()[0][0] + b))
