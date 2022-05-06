import numpy as np
import pandas as pd

def GenerateX(sentence):
    global feature #引入特征向量
    x=np.zeros(len(feature))

    #生成x
    sentence=sentence.split(' ')
    slen=len(sentence)
    for gram in range(1,4):
        if slen >= gram:
            for i in range(0,slen+1-gram):
                wordset = { word.lower() for word in sentence[i:i+gram] }
                if wordset in feature:
                    x[feature.index(wordset)]+=1
    return x

def SoftmaxRegression(sentence):
    global w #引入权重向量
    global C
    global V
    x=np.zeros(V)
    y_predict=[0]*C

    x=GenerateX(sentence)

    #计算各类别的概率
    for i in range(0,C):
        y_predict[i]= (np.array(w.loc[i])) @ (x.transpose())

    return y_predict.index(max(y_predict))

def Train_1_epoch():
    trainset=pd.read_csv('Dataset/train.tsv',sep='\t')
    global w
    global C
    global V
    global lr
    sum_matrix = np.zeros((C,V))
    trainsum=int(len(trainset['Phrase'])*0.8)

    for i in range(0,trainsum):
        y_real=np.zeros(C)
        y_real[trainset['Sentiment'][i]]=1

        x=np.zeros(V)
        x+=GenerateX(trainset['Phrase'][i])
        y_predict=np.zeros(C)
        for j in range(0,C):
            y_predict[j]= (np.array(w.loc[j])) @ (x.transpose())

        x=x.reshape(1,V)
        y_real=y_real.reshape(1,C)
        y_predict=y_predict.reshape(1,C)

        sum_matrix += ((y_real-y_predict).transpose()) @ x

    w=pd.DataFrame((np.array(w)+lr*1/trainsum*sum_matrix))

def Eval():
    testset=pd.read_csv('Dataset/train.tsv',sep='\t')
    end=len(testset['Phrase'])
    testnum=int(end*0.2)
    goal=0

    for i in range(end-testnum,end):
        if SoftmaxRegression(testset['Phrase'][i]) == testset['Sentiment'][i]:
            goal += 1

    return goal/testnum


#初始化相关参数
LIST=['Dataset/list1.csv','Dataset/list2.csv','Dataset/list3.csv'] #特征表示文件的地址

C=5            #文本分类的类别
w_init=0.1  #权重向量的初始值

lr=0.01 #learn rate 学习率

feature=[] #特征向量，即权重向量的对应分量代表的特征

for file in LIST:
    temp=pd.read_csv(file)
    for words in (temp['feat'].tolist()):
        words=str(words).split()
        feature.append({word for word in words})

V=len(feature)  #权重向量的长度
w=pd.DataFrame(np.zeros((C,V))) #权重向量表
w[:]=w_init

#！！🌟！！pandas一个细节点：初始列标签的数据类型为数字，保存再读取后类型变为字符串了
#原本用w[0]调用第一列，保存后需要用w['0']或w[str(0)]
w.to_csv('Dataset/w.csv')

w=pd.read_csv('Dataset/w.csv')
w=pd.DataFrame(np.array(w)[:,1:]) #修正存储格式

flag=1
epochs=1
score=0
score_old=-1
while flag:
    Train_1_epoch()
    score=Eval()
    print("epoch"+str(epochs)+":"+str(score))
    if score-score_old <0.01 and epochs>=10 :
        flag=0
    else:
        score_old=score
        epochs+=1

#存储学习成果
w.to_csv('Dataset/w.csv')

