from numpy import *

def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [map(float,line) for line in stringArr]
    return mat(datArr)

def pca(dataMat, topNfeat=9999999):
    meanVals = mean(dataMat, axis=0) #axis=0, 压缩行，对各列求均值，返回 1* n 矩阵
    meanRemoved = dataMat - meanVals
    covMat = cov(meanRemoved, rowvar=0)
    eigVals,eigVects = linalg.eig(mat(covMat)) #求协方差矩阵的的特征值和特征向量
    eigValInd = argsort(eigVals) #argsort返回的是数组值从小到大的索引值
    eigValInd = eigValInd[:-(topNfeat+1):-1] #取倒数前topNfeat个
    redEigVects = eigVects[:,eigValInd]
    lowDDataMat = meanRemoved * redEigVects
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    return lowDDataMat, reconMat