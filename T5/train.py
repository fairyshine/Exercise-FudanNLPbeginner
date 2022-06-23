import torch
import torch.utils.data as Data
import numpy as np

from data import getDataset,getDict
from model import PoetryGenerator
from utils import transword

file_path="Dataset/poetryFromTang.txt"

poetrys = getDataset(file_path)
word2idx = getDict(poetrys)
idx2word = { value:key for key,value in word2idx.items() }

# 定义参数

BATCH_SIZE = 32
learning_rate = 0.01
epoch_num = 10 # 100
embedding_size = 300
hidden_size = 256
dropout_size = 0.4
vocab_size = len(word2idx)
model_name = 'lstm'
num_layers = 2

# 生成数据集，用每句诗的前几个字预测最后一个字。因为每个batch的训练集长度要一致，所以五言诗和七言诗分开。
len1 = 4
len2 = 6
data = [line.replace('，', ' ').replace('。', ' ').split() for line in poetrys]

x_5, x_7, y_5, y_7 = [], [], [], []
for i in data:
    for j in i:
        if len(j) == len1+1:
            x_5.append(j[:len1])
            y_5.append(j[-1])
        elif len(j) == len2+1:
            x_7.append(j[:len2])
            y_7.append(j[-1])
        else:
            pass

x_5_vec = [transword(i,word2idx) for i in x_5]
x_7_vec = [transword(i,word2idx) for i in x_7]
y_5_vec = [transword(i,word2idx) for i in y_5]
y_7_vec = [transword(i,word2idx) for i in y_7]


# 先转换成 torch 能识别的 Dataset
torch_dataset1 = Data.TensorDataset(torch.tensor(x_5_vec, dtype=torch.long), torch.tensor(y_5_vec, dtype=torch.long))
torch_dataset2 = Data.TensorDataset(torch.tensor(x_7_vec, dtype=torch.long), torch.tensor(y_7_vec, dtype=torch.long))

# 把 dataset 放入 DataLoader
loader1 = Data.DataLoader(
    dataset=torch_dataset1,  # torch TensorDataset format
    batch_size=BATCH_SIZE,  # mini batch size
    shuffle=True
)

loader2 = Data.DataLoader(
    dataset=torch_dataset2,  # torch TensorDataset format
    batch_size=BATCH_SIZE,  # mini batch size
    shuffle=True
)



model = PoetryGenerator(vocab_size, embedding_size, hidden_size, num_layers, dropout_size, model_name)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training
for epoch in range(epoch_num):
    optimizer.zero_grad()
    for step, (batch_x, batch_y) in enumerate(loader1):
        output = model(batch_x)
        loss = criterion(output, batch_y.view(-1))
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
    for step, (batch_x, batch_y) in enumerate(loader2):
        output = model(batch_x)
        loss = criterion(output, batch_y.view(-1))
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

    loss.backward()
    optimizer.step()


# 从前n个数据随机选
def pick_top_n(preds, top_n=10):
    top_pred_prob, top_pred_label = torch.topk(preds, top_n, 1)
    top_pred_prob /= torch.sum(top_pred_prob)
    top_pred_prob = top_pred_prob.squeeze(0).detach().numpy()
    top_pred_label = top_pred_label.squeeze(0).detach().numpy()
    c = np.random.choice(top_pred_label, size=1, p=top_pred_prob)

    return c[0]

def generate_random(max_len=20):
    """自由生成一首诗歌"""
    poetry = []
    random_word = [np.random.randint(0,vocab_size)]
    input = torch.LongTensor(random_word).reshape(1,1)
    for i in range(max_len):
        # 前向计算出概率最大的当前词
        proba = model(input)
        top_index = pick_top_n(proba)
        char = idx2word[top_index]

        input = (input.data.new([top_index])).view(1, 1)
        poetry.append(char)
    return poetry

poetry = generate_random()
i = 0
for word in poetry:
    print(word,end='')
    i += 1
    if i%5==0:
        print()