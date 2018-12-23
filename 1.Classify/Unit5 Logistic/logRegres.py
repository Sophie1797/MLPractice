from math import *
from numpy import *
import matplotlib.pyplot as plt

def loadDataSet():
    dataMat = []; labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])#此处x0是偏置
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat

def sigmoid(inX):
    return 1.0/(1+exp(-inX))

#梯度上升法
def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)#用mat函数转换为矩阵之后可以才进行一些线性代数的操作
    labelMat = mat(classLabels).transpose()#转置成一列
    m,n = shape(dataMatrix)
    alpha = 0.001#向目标移动的步长
    maxCycles = 500
    weights = ones((n,1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights

def stocGradAscent0(dataMatrix, classLabels, numIter=150):
    m,n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    for k in range(numIter):
        for i in range(m):
            s = dataMatrix[i]
            test = dataMatrix[i]*weights
            h = sigmoid(dataMatrix[i]*weights)
            error = classLabels[i] - h
            weights = weights + alpha * error * dataMatrix[i]
    return weights

def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m,n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    for i in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.01
            randIndex = int(random.uniform(0,len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights

def plotBestFit(weights):
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)#在一张figure里面生成多张子图,eg.参数349：将画布分割成3行4列，图像画在从左到右从上到下的第9块,供111参考
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0,3.0,0.1)
    print(type(x))
    y = (-float(weights[0])-x*float(weights[1]))/float(weights[2])#0=w0x0+w1x1+w2x2,x0=1
    #weight是一个matrix，得把它里面的元素变成普通的数字才能这样运算，否则就是矩阵运算，（1,60）
    print(type(y))
    ax.plot(x,y)
    plt.xlabel('X1'); plt.ylabel('X2')
    plt.show()

dataArr, labelMat = loadDataSet()
weights = stocGradAscent0(dataArr, labelMat)
plotBestFit(weights)