import re
from numpy import  *

def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]
    return postingList,classVec

def createVocabList(dataSet):#把所有各异的词提出来
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]=1
        else:
            print("the word: %s is not in mu Vocabulary!" % word)
    return returnVec

def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Denom = 2.0#????????????????
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num / p1Denom)
    p0Vect = log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)#vec2Classify*p1Vec的目的是只留下出现的那些词概率，p(w1,w2,w3|c1),p(w2,w6|c1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1>p0:
        return 1
    else:
        return 0

def test():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    print(myVocabList)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    print(array(trainMat))
    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))
    print(p0V)
    print(p1V)
    print(pAb)
    testEntry = ['love','my','dalmation','english']
    testDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(testDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage', 'bear']
    testDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(testDoc, p0V, p1V, pAb))

def textParse(bigString):
    listOfTokens = re.split(r'\W*', bigString)#表示分隔符是除了单词数字外的任意字符，由此得到所有的单词
    return [tok.lower() for tok in listOfTokens if len(tok)>2]

def spamTest():
    docList=[]; classList=[]; fullText=[]
    for i in range(1,26):#此处是提前知道了有多少个文件否则这样的写法。。。
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet = list(range(50)); testSet=[]
    #生成测试集
    for i in range(10):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat=[];trainClasses=[]
    #剩下的40个用来训练
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount=0
    for index in testSet:
        wordVector = setOfWords2Vec(vocabList, docList[index])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[index]:
            errorCount += 1
    print('the error rate is: ', float(errorCount)/len(testSet))

spamTest()