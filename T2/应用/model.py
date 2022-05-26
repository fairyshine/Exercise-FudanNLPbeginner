import torch
import torch.nn as nn
import torch.nn.functional as F

category=173

channel=1

class TextCNN(nn.Module):
    def __init__(self,wordNum,padding_idx):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(wordNum,5,padding_idx=padding_idx)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1,channel,(k,5)) for k in (2,3,4)])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(channel*3, category)

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