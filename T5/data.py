
def getDataset(file_path):
    # 读取文件
    poetrys = []
    poetry = ''
    with open(file_path, encoding='utf-8') as f:
        for line in f:
            if len(line)!=1:
                poetry += line.strip('\n')
            else:
                poetrys.append(poetry)
                poetry = ''
    return poetrys

def getDict(dataset):
    poetrys=dataset
    # 生成词库
    all_word = ''
    for potery in poetrys:
        all_word += potery

    all_word = all_word.replace('，','').replace('。','')

    # 统计词频
    word_dict = {}
    for word in all_word:
        if word not in word_dict:
            word_dict[word] = 1
        else:
            word_dict[word] += 1

    word_sort = sorted(word_dict.items(),key=lambda x:x[1],reverse=True)

    word2idx={}

    for i in range(len(word_sort)):
        word2idx[word_sort[i][0]]=i;

    return word2idx

