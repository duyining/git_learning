'''
@Author: 一蓑烟雨任平生
@Date: 2020-01-06 10:48:50
@LastEditTime : 2020-01-07 12:30:11
@LastEditors  : Please set LastEditors
@Description: 手写逻辑回归，逻辑回归是使用极大似然估计来实现参数的更新的！极大似然估计是 给定结果去反推参数的一种方法,目前遇到的问题：什么时候令偏导等于0？什么时候使用梯度下降？
@FilePath: /machine_learning_by_hand/logistic.py
'''

import numpy as np 
import math 
from sklearn.datasets import load_iris


class logistic():
    def loaddata(self):
        dataArr = []
        labelArr = []
        iris = load_iris()
        label = iris['target']
        for item in label:
            if item == 0:
                labelArr.append(1)
            elif item == 1:
                labelArr.append(-1)
        dataArr = iris.data[0:100]
        return dataArr, labelArr

    def sigmoid(self,inX):
        '''
        @description:实现sigmod函数 
        @param xxx
        @return: 经过sigmod变换后的函数值
        '''
        return 1.0/(1 + np.exp(-inX))

    def gradAscent(self, dataMatIn, classLabels):
        '''
        @description:批量梯度下降法 
        @param dataMatIn: 输入的数组
        @param classLabels : 数组对应的标签
        @return: 模型的参数
        '''
        dataMatrix = np.mat(dataMatIn)
        labelMatrix = np.mat(classLabels).transpose()
        m, n = np.shape(dataMatrix)
        alpha = 0.01
        maxCycles = 5
        weights = np.ones((n, 1))
        for k in range(maxCycles):
            h = self.sigmoid(dataMatrix * weights)
            error = (labelMatrix - h)
            #矩阵的乘法实现对多个数组的求和
            weights += alpha * dataMatrix.transpose() * error
        return weights

    def stocGradAscent0(self, dataMatrix, classLabels):
        '''
        @description: 随机梯度下降算法 
        @param 同批量梯度下降
        @return: 模型参数
        '''    
        #在实现上与批量梯度下降的区别：变量h与error都是数值，而批量则都是向量。随机梯度没有矩阵转化的过程，所有变量的数据类型都是数组
        m, n = np.shape(dataMatrix)
        alpha = 0.01
        weights = np.ones(n)
        for i in range(m):
            h = self.sigmoid(sum(dataMatrix[i] * weights))
            error = classLabels[i] - h 
            weights = weights + alpha * error * dataMatrix[i]
        return weights
    
    def train_logistic(self):
        dataArr,labelArr = self.loaddata()
        pass
    
if __name__ == "__main__":
    A = logistic()
    dataArr, labelArr = A.loaddata()
    res = A.gradAscent(dataArr, labelArr)
    print(res)
