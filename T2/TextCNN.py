import json

import torch
import torch.nn as nn
import torch.nn.functional as F


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

train_data_tensor=torch.zeros((140454,1,60,5))
train_target_tensor=torch.zeros((140454))

test_data_tensor=torch.zeros((15606,1,60,5))
test_target_tensor=torch.zeros((15606))

embeds = nn.Embedding(19479, 5)

for i in range(0,len(train_list)):
    phrase=train_list[i]["Phrase"].split(' ');
    for j in range(0,len(phrase)):
        train_data_tensor[i][0][j]=embeds(torch.LongTensor([word_To_num[phrase[j].lower()]]))
    train_target_tensor[i]=train_list[i]["Sentiment"]

for i in range(0,len(test_list)):
    phrase=test_list[i]["Phrase"].split(' ');
    for j in range(0,len(phrase)):
        test_data_tensor[i][0][j]=embeds(torch.LongTensor([word_To_num[phrase[j].lower()]]))
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
        self.convs = nn.ModuleList(
            [nn.Conv2d(1,16,(k,5)) for k in (2,3,4)])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(16*3, 5)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out = torch.cat([self.conv_and_pool(x, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out

net = TextCNN()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(10):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(train_loader, start=0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        print('Step ',i,' Completed!');

        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

    correct=0
    total=0
    for data in test_loader:
        inputs,labels=data
        outputs=net(inputs)
        _,predicted=torch.max(outputs.data,1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    print("Epoch:",epoch,'Accuracy:',1.0*correct/total)

print('Finished Training')

PATH = './TextCNN.pth'
torch.save(net.state_dict(), PATH)