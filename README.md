

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

# Task 2 

# Task 3

# Task 4

# Task 5
