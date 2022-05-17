

# Exercise-FudanNLPbeginner

复旦邱锡鹏老师的自然语言处理入门练习作业



## Task 1 NumPy文本分类

多分类问题（分成5类）C=5

数据集：Kaggle 影评情感分析  

流程：

（1）特征表示：tri-gram的词袋模型

得到一个列表，长度V

以单词为单位，抽取数据集文本中包含的所有一元集、二元相邻集、三元相邻集

（2）学习器：logistic回归（二分类）—>  softmax回归（多分类）

$$
\hat{y} =\arg \max^{C}_{c=1} \omega^{T}_{c} x
$$

omega，5个权重向量，长度V

（3）训练：交叉熵损失

评估损失情况：风险函数（交叉熵损失），评分，判断是否继续

调参：-1 * 风险函数的梯度 * 学习率

---

数据集：

自行下载原数据集存储至Dataset/train.tsv、Dataset/test.tsv

---

特征表示.ipynb

处理原数据集内容，获取特征，保存至list1.csv，list2.csv，list3.csv

Softmax回归.ipynb

实现基于numpy的softmax回归，写得较一般 0.0

---

可作实验：

shuffle ：打乱序列里的元素，随机排列

参数更新：batch  遍历全部数据集算一次Loss

​					mini-batch  数据集分拆成小批量，按小批量更新参数

---

改进：

1.数据集的存取方式，pandas.DataFrame有些细节没处理好

2.数据集的处理，可以把训练集和测试集分清

3.主要函数的参数输入没弄好，不方便单条数据测试

4.函数运行效率优化（特征向量采用字典？）

由于之后主要采用各类框架，任务1不准备作进一步优化了。

# Task 2 

PyTorch重写Task1

流程：

（1）词嵌入 Word Embedding：使用word2vec？

文本表示的类型：

- 基于one-hot、tf-idf、textrank等的bag-of-words；
- 主题模型：LSA（SVD）、pLSA、LDA；
- 基于词向量的固定表征：word2vec、fastText、glove
- 基于词向量的动态表征：ELMO、GPT、bert

（2）CNN/RNN的分类器

（3）训练

---

### Dataprocess.ipynb 处理数据集

#### 1 构造 word embedding字典

1.1

根据编号，train.tsv和test.tsv 共222352条phrase

将这些短句分割为单词，全部字母小写化，得到19479个词(或标点)。

根据出现频率倒序储存在字典列表wordFreq中，保存于文件Dateset/wordFreq.jsonl

1.2

将word embedding字典存储，word<—>序号

保存于文件Dataset/word_To_num.json、Dataset/num_To_word.json

#### 2 转换成jsonl格式，分割数据集

2.1

原格式：tsv 有PhraseId、SentenceId、Phrase、Sentiment

现格式： (使用pandas库，这里要先把几个数字转化为字符串储存，在单独转化为数字)

```json
{
  "PhraseId":, //超过int16范围，转换为字符串
  "SentenceId":,
  "Phrase":" ",
  "Sentiment": //0-4间的整数，2无情感，4为喜欢，0为讨厌
}
```

2.2

短句共156060条

0-140453 为训练集 ，保存为 Dataset/train.jsonl

140454-156059 为测试集，保存为 Dataset/test.jsons

> 两数据集连接处可能有短句相叠，不够严谨，但无伤大雅。

---

### TextCNN.pynb 

```python
import json

max=0
with open('Dataset/Dataset.jsonl','r') as f:
        for line in f:
            data=json.loads(line)
            temp=len(data['Phrase'].split(' '))
            if temp>max:
                max=temp;
print(max)
```

 据分析：Phrase最大句长为52，故设置矩阵高度为60



# Task 3

# Task 4

# Task 5
