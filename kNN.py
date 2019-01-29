from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels =['A','A','B','B']
    return group,labels

def classify0(inX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX,(dataSetSize,1)) - dataSet #tile将inX二维数组化，变成dataSetSize行1列
    sqDiffMat = diffMat**2 #各个元素分别平方
    sqDistances = sqDiffMat.sum(axis=1)#axis=1,表示矩阵中行之间数的求和；axis=0,表示矩阵中对列求和
    distances = sqDistances**0.5 #开根得到距离
    sortedDistIndicies = distances.argsort()#argsort表示升序排列,输出对应的索引
    classCount={} #初始化classCount
    for i in range(k): #选择距离最小的k个点
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1 #检测并生成新元素，0只做初始化作用，计数，遇到相同的加1
    sortedClassCount = sorted(classCount.items(),
                              key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

###从文本文件中解析数据
def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines) #文件的行数
    returnMat = zeros((numberOfLines,3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip() #截取掉所有的回车字符
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3] #取前三个元素存储到特征矩阵中
        if listFromLine[-1] == 'didntLike':
            classLabelVector.append(1)
        elif listFromLine[-1] == 'smallDoses':
            classLabelVector.append(2)
        elif listFromLine[-1] == 'largeDoses':
            classLabelVector.append(3)
        #classLabelVector.append(int(listFromLine[-1]))#索引值-1表示列表中的最后一列元素
        index += 1
    return returnMat,classLabelVector

###归一化特征值
def autoNorm(dataSet):
    minVals = dataSet.min(0) #对dataset数据集的列求最小值
    maxVals = dataSet.max(0) 
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0] #第一维的长度，矩阵行数
    normDataSet = dataSet - tile(minVals,(m,1))
    normDataSet = normDataSet/tile(ranges,(m,1))
    return normDataSet, ranges, minVals

###分类器针对约会网站的测试代码
def datingClassTest():
    hoRatio = 0.10
    datingDataMat,datingLabels = file2matrix('datingTestSet.txt') #处理读取文件数据
    normMat,ranges,minVals = autoNorm(datingDataMat) #归一化特征值
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio) #测试集个数
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],6)
        #分类器执行处理
        print("the classifier came back with: %d, the real answer is: %d" %(classifierResult,datingLabels[i]))
        if(classifierResult != datingLabels[i]):
            errorCount += 1.0
    print("the total error rate is: %f" %(errorCount/float(numTestVecs)))

###约会网站预测函数
def classifyPerson():
    resultList = ['not at all','in small doses','in large doses']
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat,datingLabels = file2matrix('datingTestSet.txt')
    normMat,ranges,minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles,percentTats,iceCream])
    classifierResult = classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
    print("You will probably like this person:",resultList[classifierResult - 1])


def main():
##    (group,labels) = createDataSet()
##    print ('[0.4,0.1]','predict:',classify0([0.4,0.1],group,labels,3))
##    print ('[1.4,1.1]','predict:',classify0([1.4,1.1],group,labels,3))    
    datingDataMat,datingLabels = file2matrix('datingTestSet.txt')
    print(datingDataMat)
    print(datingLabels[0:20])
##使用matplotlib创建散点图
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    plt.xlabel("%time")
    plt.ylabel("L")
    ax1.scatter(datingDataMat[:,1],datingDataMat[:,2],
               15.0*array(datingLabels),15.0*array(datingLabels))
    ax2 = fig.add_subplot(212)
    plt.xlabel("miles")
    plt.ylabel("%time")
    ax2.scatter(datingDataMat[:,0],datingDataMat[:,1],
               15.0*array(datingLabels),15.0*array(datingLabels))
    plt.show()
##执行autoNorm(datingDataMat)函数
    normMat,ranges,minVals = autoNorm(datingDataMat)
##    print(normMat)
##    print(ranges)
##    print(minVals)
##   datingClassTest()
    classifyPerson()
    
    
if __name__=="__main__":
    main()





























