import torch
import torch.nn as nn

class PoetryGenerator(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, dropout_size, model_name='lstm'):
        super(PoetryGenerator, self).__init__()
        self.model = model_name
        self.embed = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, batch_first=True, dropout=dropout_size)
        self.gru = nn.GRU(embedding_size, hidden_size, num_layers, batch_first=True, dropout=dropout_size )
        self.F = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x_embedding = self.embed(x)
        if self.model=='lstm':
            out, (hn,cn) = self.lstm(x_embedding)
        else:
            out, (hn,cn) = self.gru(x_embedding)
        #print("RNN网络的参数：",hn)
        outputs = self.F(out[:,-1,:])

        return outputs