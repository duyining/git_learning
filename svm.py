from numpy import *
import math
import json 


def selectJrand(i,m):
    j=i #we want to select any J not equal to i
    while (j==i):
        j = int(random.uniform(0,m))
    return j

def clipAlpha(aj,H,L):
    if aj > H: 
        aj = H
    if L > aj:
        aj = L
    return aj

def kernelTrans(X, A, kTup): #calc the kernel or transform data to a higher dimensional space
    m,n = shape(X)
    K = mat(zeros((m,1)))
    if kTup[0]=='lin': 
        K = X * A.T   #linear kernel
    elif kTup[0]=='rbf':
        for j in range(m):
            deltaRow = X[j,:] - A
            K[j] = deltaRow*deltaRow.T
        K = exp(K/(-1*kTup[1]**2)) #divide in NumPy is element-wise not matrix like Matlab
    else: raise NameError('Houston We Have a Problem -- \
    That Kernel is not recognized')
    return K


class optStruct:
    def __init__(self,dataMatIn, classLabels, C, toler, kTup):  # Initialize the structure with the parameters 
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C # C是超参数
        self.tol = toler # toler是误差项，当误差小于这个值时，可以停止循环
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m,1)))
        self.b = 0
        self.eCache = mat(zeros((self.m,2))) # 误差缓存，第一列给出的是eCache是否有效的标志位，第二列给出的是实际的E值。
        self.K = mat(zeros((self.m,self.m))) # 计算核函数.
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.X, self.X[i,:], kTup) # 对变量使用核函数，self.X表示输入的矩阵，self.X[i,:]表示矩阵的第i行.self.K[:,i]表示K矩阵中的第i列


def calcEk(oS, k):
    '''
    @description: 计算误差的函数，计算E值并返回。
    @param oS:class, k：int 
    @return: float
    '''
    fXk = float(multiply(oS.alphas,oS.labelMat).T*oS.K[:,k] + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek


def selectJ(i, oS, Ei):         
    '''
    @description: 内循环中的启发式方法，用来选择第二个alpha，并计算E
    @param i : 第一个选择的alpha的下标，Ei第一个选择的alpha计算出来的误差
    @return: 
    '''
    maxK = -1; maxDeltaE = 0; Ej = 0
    oS.eCache[i] = [1,Ei]  #set valid #choose the alpha that gives the maximum delta E
    validEcacheList = nonzero(oS.eCache[:,0].A)[0]
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:   #loop through valid Ecache values and find the one that maximizes delta E
            if k == i: 
                continue #don't calc for i, waste of time
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k; maxDeltaE = deltaE; Ej = Ek
        return maxK, Ej
    else:   #如果这是第一次循环的话，那么就随机选择一个J值
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej


def updateEk(oS, k):
    '''
    @description: 每个alpha改变过之后，更新缓存。
    @param {type} 
    @return: 
    '''
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1,Ek]


def innerL(i, oS):
    '''
    @description: 完整的内循环方法 
    @param {type} 
    @return: 
    '''
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
        j,Ej = selectJ(i, oS, Ei) #使用启发式方法选择第二个alpha
        alphaIold = oS.alphas[i].copy(); alphaJold = oS.alphas[j].copy();
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L==H: print ("L==H"); return 0
        eta = 2.0 * oS.K[i,j] - oS.K[i,i] - oS.K[j,j] #计算eta的方法改为使用核函数
        if eta >= 0: 
            print ("eta>=0"); 
            return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
        updateEk(oS, j) #更新缓存
        if (abs(oS.alphas[j] - alphaJold) < 0.00001): 
            print ("j not moving enough")
            return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])#根据第二个变量更新第一个变量
        updateEk(oS, i) #更新缓存
        b1 = oS.b - Ei- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,i] - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[i,j]
        b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,j]- oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[j,j]
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): 
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]): 
            oS.b = b2
        else: 
            oS.b = (b1 + b2)/2.0
        return 1
    else: 
        return 0


def smoP(dataMatIn, classLabels, C, toler, maxIter,kTup=('rbf', 1.3)):    #full Platt SMO
    '''
    @description: 完整的外循环方法，选择的方法是违反kkt条件最严重的一个。
    @param {type} 
    @return: 
    '''
    oS = optStruct(mat(dataMatIn),mat(classLabels).transpose(),C,toler, kTup)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        #当循环达到最大值或者遍历整个集合未对alpha做出修改，退出循环
        alphaPairsChanged = 0
        if entireSet:#遍历所有值
            for i in range(oS.m):        
                alphaPairsChanged += innerL(i,oS)
                print ("fullSet, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
            iter += 1
        else:#遍历alpha边界值，也就是说，遍历所有的支持向量
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0] #取出所有的支持向量对应的alpha
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i,oS)
                print ("non-bound, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
            iter += 1
        if entireSet: 
            entireSet = False #toggle entire set loop
        elif (alphaPairsChanged == 0): 
            entireSet = True  
        print ("iteration number: %d" % iter)
    return oS.b,oS.alphas


def calcWs(alphas,dataArr,classLabels):
    X = mat(dataArr); labelMat = mat(classLabels).transpose()
    m,n = shape(X)
    w = zeros((n,1))
    for i in range(m):
        w += multiply(alphas[i]*labelMat[i],X[i,:].T)
    return w

from sklearn.datasets import load_iris

def testRbf(k1=1.3):
    # dataArr,labelArr = loadDataSet('testSetRBF.txt')
    dataArr = []
    labelArr = []
    with open('/Users/duyining/Desktop/neg_base_vec.json') as file :
        neg_data = list(json.loads(file.read()).values())
    with open('/Users/duyining/Desktop/pos_base_vec.json') as file :
        pos_data = list(json.loads(file.read()).values())
    neg_label = [-1 for _ in range(len(neg_data))]
    pos_label = [1 for _ in range(len(pos_data))]
    dataArr = neg_data + pos_data
    labelArr = neg_label + pos_label
    # iris = load_iris()
    # label = iris['target']
    # for item in label:
    #     if item == 0:
    #         labelArr.append(1)
    #     elif item == 1:
    #         labelArr.append(-1)
    # # dataArr = stack((iris.data[:,0],iris.data[:,1]),axis = 1)[0:100]
    # dataArr = iris.data[0:100]
    dataArr, datatest,labelArr, labeltest, _ = split_data(dataArr, labelArr)
    b,alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1)) #C=200 important
    datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
    svInd=nonzero(alphas.A>0)[0]
    sVs=datMat[svInd] #get matrix of only support vectors
    labelSV = labelMat[svInd];
    print ("支持向量的个数:{}".format(shape(sVs)[0]))
    m,n = shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],('rbf', k1))
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1
    print("在训练数据上的错误率:{}".format(float(errorCount)/m))
    errorCount = 0
    datMat=mat(datatest); labelMat = mat(labeltest).transpose()
    m,n = shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],('rbf', k1))
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1    
    print("在测试数据上的错误率:{}".format(float(errorCount)/m))

from sklearn.utils import shuffle

def split_data(X, y, test_size = 0.2, random_state=42):
    X, y = array(X), array(y)
    index = list(range(len(X)))
    index_shuffle = shuffle(index, random_state=random_state)
    X_shuffle, y_shuffle = X[index_shuffle], y[index_shuffle]
    X_test_size = int(len(X)*test_size)
    X_test, y_test = X_shuffle[:X_test_size], y_shuffle[:X_test_size]
    X_train, y_train = X_shuffle[X_test_size:], y_shuffle[X_test_size:]
    return X_train, X_test, y_train, y_test, index_shuffle

if __name__ == "__main__":
    testRbf()
