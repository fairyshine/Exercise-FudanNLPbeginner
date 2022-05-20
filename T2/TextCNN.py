import json

import torch
import torch.nn as nn
import torch.nn.functional as F

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#print(device)

word_To_num={}
num_To_word={}


with open('Dataset/word_To_num.json','r') as f:
    word_To_num=json.load(f)

with open('Dataset/num_To_word.json','r') as f:
    num_To_word=json.load(f)

train_list=[]
test_list=[]

with open('Dataset/train.jsonl','r') as f:
        for line in f:
            train_list.append(json.loads(line))

with open('Dataset/test.jsonl','r') as f:
        for line in f:
            test_list.append(json.loads(line))

train_data_tensor=torch.zeros((140454,60))
train_target_tensor=torch.zeros((140454))

test_data_tensor=torch.zeros((15606,60))
test_target_tensor=torch.zeros((15606))

for i in range(0,len(train_list)):
    phrase=train_list[i]["Phrase"].split(' ');
    for j in range(0,len(phrase)):
        train_data_tensor[i][j]=torch.LongTensor([word_To_num[phrase[j].lower()]])
    for j in range(len(phrase),60):
        train_data_tensor[i][j]=torch.LongTensor([19478])
    train_target_tensor[i]=train_list[i]["Sentiment"]

for i in range(0,len(test_list)):
    phrase=test_list[i]["Phrase"].split(' ');
    for j in range(0,len(phrase)):
        test_data_tensor[i][j]=torch.LongTensor([word_To_num[phrase[j].lower()]])
    for j in range(len(phrase),60):
        test_data_tensor[i][j]=torch.LongTensor([19478])
    test_target_tensor[i]=test_list[i]["Sentiment"]

train_target_tensor=train_target_tensor.long()
test_target_tensor=test_target_tensor.long()

train_dataset=torch.utils.data.TensorDataset(train_data_tensor, train_target_tensor)
test_dataset=torch.utils.data.TensorDataset(test_data_tensor, test_target_tensor)

train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=100,shuffle=True) #shuffle会在每个epoch里打乱顺序
test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=100,shuffle=True)

class TextCNN(nn.Module):
    def __init__(self):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(19479,5)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1,1,(k,5)) for k in (2,3,4)])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(1*3, 5)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out = self.embedding(x.long())
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out

net = TextCNN()
#net = net.to(device)

criterion = nn.CrossEntropyLoss()
#criterion = criterion.to(device)
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(500):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader, start=0):
        #inputs=inputs.to(device)
        #labels=labels.to(device)

        outputs = net(inputs)
        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        print('Step ',i,' Completed!  loss this time: ',loss.item());

        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # print every 100 mini-batches
            print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

    with torch.no_grad():
        correct=0
        total=0
        net.eval()
        for (inputs,labels) in test_loader:
            #inputs=inputs.to(device)
            #labels=labels.to(device)
            outputs=net(inputs)
            _,predicted=torch.max(outputs.data,1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        print("Epoch:",epoch,'Accuracy:',1.0*correct/total)

print('Finished Training')

PATH = './TextCNN.pth'
torch.save(net.state_dict(), PATH)