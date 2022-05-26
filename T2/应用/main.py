import torch
import torch.nn as nn

from dataprocess import getDataset
from model import TextCNN


#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#print(device)

trainDataset_path='../data/data_divide/train.jsonl'
testDataset_path='../data/data_divide/test.jsonl'


train_dataset,test_dataset,wordNum,padding_idx=getDataset(trainDataset_path,testDataset_path)

train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=100,shuffle=True) #shuffle会在每个epoch里打乱顺序
test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=100,shuffle=True)

net = TextCNN(wordNum,padding_idx)
#net = net.to(device)

criterion = nn.CrossEntropyLoss()
#criterion = criterion.to(device)
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(500):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader, start=0):
        #inputs=inputs.to(device)
        #labels=labels.to(device)

        net.train()

        outputs = net(inputs)
        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        print('Step ',i,' Completed!  loss this time: ',loss.item()); #查看实时状态

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