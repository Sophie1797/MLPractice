from numpy import *
import matplotlib
import matplotlib.pyplot as plt
import operator

def file2matrix(filename):
    fr = open(filename)
    arrayOfLines = fr.readlines()
    numberOfLines = len(arrayOfLines)
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOfLines:
        line = line.strip() # trim all \r\n
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector

def drawFigure(datingDataMat, datingLabels):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1], 15.0 * array(datingLabels), 15.0 * array(datingLabels))
    # 加上后两个参数的话，可以让同样标签的点一个颜色，一样大，而且标签值越大，点越大
    plt.show()

def autoNorm(dataSet):
    minVals = dataSet.min(0) # min value of every column (row 1)
    maxVals = dataSet.max(0) # max value of every column
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0] # 4*2, shape(0)=4,shape(1)=2, (no.x dim)
    normDataSet = dataSet - tile(minVals, (m, 1)) # tile: manage input array param repeat as (m,1)
    normDataSet = normDataSet/tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis = 1) # every row
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort() # sort and return index
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1 #if dict[i] not exist, new and set 0
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0]

def datingClassTest():
    hoRatio = 0.1 # test line num ratio
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("the  classifier came back with: %d, the real answer is : %d", classifierResult, datingLabels[i])
        if(classifierResult != datingLabels[i]):
            errorCount += 1.0
    print("the total error rate is: %f", errorCount/float(numTestVecs))

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],90,0.1])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

