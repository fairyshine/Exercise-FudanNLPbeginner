from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
progress = Progress(
    TextColumn("[bold blue]运行中...", justify="right"),
    BarColumn(bar_width=None),
    "[progress.percentage]{task.percentage:>3.1f}%",
    "•",
    TimeElapsedColumn(),
    "•",
    TimeRemainingColumn(),
)

import torch
import torch.optim as optim

from data import getDataset
from model import BiLSTM_CRF
from utils import read_json,prepare_sequence

torch.manual_seed(1)

Train_Path='Dataset/train.jsonl'
Test_Path='Dataset/test.jsonl'
Word2idx_Path='Dataset/word2idx.json'
Tag2idx_Path='Dataset/tag2idx.json'

EMBEDDING_DIM = 5
HIDDEN_DIM = 4
BATCH_SIZE=100

trainDataset = getDataset(Train_Path)
testDataset = getDataset(Test_Path)

word_to_ix = read_json(Word2idx_Path)
tag_to_ix = read_json(Tag2idx_Path)

model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)


# Check predictions before training
with torch.no_grad():
    precheck_sent = prepare_sequence(trainDataset[0][0], word_to_ix)
    #precheck_tags = torch.tensor([tag_to_ix[t] for t in trainDataset[0][1]], dtype=torch.long)
    print(model(precheck_sent),'试运行成功！')

# Make sure prepare_sequence from earlier in the LSTM section is loaded
with progress:
    for epoch in progress.track(range(30)):  # again, normally you would NOT do 300 epochs, it is toy data
        for sentence, tags in trainDataset:
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Step 2. Get our inputs ready for the network, that is,
            # turn them into Tensors of word indices.
            sentence_in = prepare_sequence(sentence, word_to_ix)
            targets = prepare_sequence(tags, tag_to_ix)

            # Step 3. Run our forward pass.
            loss = model.neg_log_likelihood(sentence_in, targets)

            # Step 4. Compute the loss, gradients, and update the parameters by
            # calling optimizer.step()
            loss.backward()
            optimizer.step()


# Check predictions after training
score=0
with torch.no_grad():
    for sentence, tags in testDataset:
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = prepare_sequence(tags, tag_to_ix).numpy().tolist()
        __ , result=model(sentence_in)
        print(result)
        print(targets)
        if result == targets:
            score+=1
    print("score:",score/len(testDataset))
# We got it!