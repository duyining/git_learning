'''
@Author: 一蓑烟雨任平生
@Date: 2020-01-07 14:54:53
@LastEditTime : 2020-01-07 15:37:28
@LastEditors  : Please set LastEditors
@Description: 手写随机梯度和批量梯度下降算法，并尝试给出随机梯度算法合理的解释
随机梯度每次只是用一个数据进行训练，所以每次的训练速度快，但是并不一定是沿着梯度下降最快的方向走的。所以迭代次数会更多
批量梯度每次迭代都会使用全部的数据集，所以一次训练的时间要长一点。但是迭代的次数会较少。在数据量较大时，一般使用随机梯度
@FilePath: /machine_learning_by_hand/ascent.py
'''

import numpy as np 
import random

class Ascent():
    
    def gradAscent(self,x, y, theta, alpha, m, maxIterations):
        '''
        @description:批量梯度下降 
        @param {type} 
        @return: 
        '''
        xTrains = x.transpose()
        count = 0 
        for i in range(0, maxIterations):
            hypothesis = np.dot(x, theta)
            loss = hypothesis - y 
            gradient = np.dot(xTrains, loss)/m
            if((np.zeros(np.shape(gradient)) == gradient).any()):
                count += 1 
            theta = theta - alpha * gradient
        print('批量梯度经过{}次收敛'.format(count))
        return theta
    
    def stocGradAscent(self, x, y, theta, alpha, m, maxIterations):
        '''
        @description:随机梯度下降 
        @param {type} 
        @return: 
        '''
        data = []
        for i in range(10):
            data.append(i)
        xTrains = x.transpose()
        count = 0
        for i in range(0, maxIterations):
            hypothesis = np.dot(x, theta)
            loss = hypothesis - y 
            index = random.sample(data, 1)
            index1 = index[0]
            gradient = loss[index] * x[index1]
            if((np.zeros(np.shape(gradient)) == gradient).all()):
                count += 1 
            theta = theta - alpha * gradient.T
        print('随机梯度经过{}次收敛'.format(count))
        return theta
        

    def predict(self, x, theta):
        '''
        @description: 使用训练好的模型预测数据
        @param {type} 
        @return: 
        '''
        m, n = np.shape(x)
        xTest = np.ones((m, n+1))
        xTest[:, :-1] = x  
        res = np.dot(xTest, theta)
        return res 

if __name__ == "__main__":
    print('start...')
    A = Ascent()
    trainData = np.array([[1.1,1.5,1],[1.3,1.9,1],[1.5,2.3,1],[1.7,2.7,1],[1.9,3.1,1],[2.1,3.5,1],[2.3,3.9,1],[2.5,4.3,1],[2.7,4.7,1],[2.9,5.1,1]])
    trainLabel = np.array([2.5,3.2,3.9,4.6,5.3,6,6.7,7.4,8.1,8.8])
    m, n = np.shape(trainData)
    theta = np.ones(n)
    alpha = 0.1
    maxIteration = 5000
    #下面返回的theta就是学到的theta
    theta = A.gradAscent(trainData, trainLabel, theta, alpha, m, maxIteration)
    print ("theta = ",theta)
    x = np.array([[3.1, 5.5], [3.3, 5.9], [3.5, 6.3], [3.7, 6.7], [3.9, 7.1]])
    print (A.predict(x, theta))
    theta = A.stocGradAscent(trainData, trainLabel, theta, alpha, m, maxIteration)
    print ("theta = ",theta)
    x = np.array([[3.1, 5.5], [3.3, 5.9], [3.5, 6.3], [3.7, 6.7], [3.9, 7.1]])
    print (A.predict(x, theta))
    #yes,is the code