import json
import torch
from gensim import corpora

def getDataset(train_path,test_path):
    train_list=[]
    test_list=[]
    #======读取文件======
    with open(train_path,'r') as f:
        for line in f:
            train_list.append(json.loads(line))
    with open(test_path,'r') as f:
        for line in f:
            test_list.append(json.loads(line))

    with open('../data/data_divide/label2id.json','r') as f:
        label2id=json.load(f)

    #====== ======
    text_corpus=[]
    maxlen=0
    for data in train_list:
        datalist=[t for t in data['text']]
        text_corpus.append(datalist)
        if len(datalist)>maxlen:
            maxlen=len(datalist)
    for data in test_list:
        datalist=[t for t in data['text']]
        text_corpus.append(datalist)
        if len(datalist)>maxlen:
            maxlen=len(datalist)

    dictionary = corpora.Dictionary(text_corpus)
    dic=dictionary.token2id
    space=dic[' ']

    maxlen=(maxlen//10+1)*10

    #====== ======
    train_data_tensor=[[]]*len(train_list)
    train_target_tensor=[0]*len(train_list)
    test_data_tensor=[[]]*len(test_list)
    test_target_tensor=[0]*len(test_list)

    for i in range(len(train_list)):
        data=train_list[i]
        textvector=[space]*maxlen
        for j in range(len(data['text'])):
            textvector[j]=dic[data['text'][j]]
        train_data_tensor[i]=textvector
        train_target_tensor[i]=label2id[data['eventTags'][0]['eventType']]

    for i in range(len(test_list)):
        data=test_list[i]
        textvector=[space]*maxlen
        for j in range(len(data['text'])):
            textvector[j]=dic[data['text'][j]]
        test_data_tensor[i]=textvector
        test_target_tensor[i]=label2id[data['eventTags'][0]['eventType']]

    train_data_tensor=torch.LongTensor(train_data_tensor)
    train_target_tensor=torch.LongTensor(train_target_tensor)
    test_data_tensor=torch.LongTensor(test_data_tensor)
    test_target_tensor=torch.LongTensor(test_target_tensor)

    train_dataset=torch.utils.data.TensorDataset(train_data_tensor, train_target_tensor)
    test_dataset=torch.utils.data.TensorDataset(test_data_tensor, test_target_tensor)

    return train_dataset,test_dataset,len(dic),space