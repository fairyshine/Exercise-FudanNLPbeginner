
# 将字序列转化为id序列
def transword(char_list,word2idx):
    ids = [word2idx[char] for char in char_list]
    return ids