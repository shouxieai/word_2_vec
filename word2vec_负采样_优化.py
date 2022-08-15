import pickle
import jieba
import pandas as pd
from tqdm import tqdm
import numpy as np


# 当前词语 预测其他词语
def clean_info(X):  # 需要传入X和y, 因为文本经过处理之后,有可能为空, 需要删除为空的文本, 此时应该删除对应的y
    result_x = []
    stop_words = load_stop_word() + ["\u3000"]

    for index, content in tqdm(enumerate(X)):
                result_x.append([i for i in jieba.lcut(content) if i not in stop_words])
    return result_x

def build_word_dict(contents_list):
    word_2_index = {}
    index_2_word = {}

    n = 0
    for content in contents_list:
        for word in content:
            if word not in word_2_index:
                word_2_index[word] = word_2_index.get(word,n)
                n+=1

    one_hot_result = []
    for index,key in enumerate(word_2_index):
        one_hot = [0 for _ in range(len(word_2_index))]
        one_hot[index] = 1
        one_hot_result.append(one_hot)
        index_2_word[index] = key
    return word_2_index,index_2_word,np.array(one_hot_result,dtype=np.int32).reshape(len(one_hot_result),1,-1)

def load_stop_word(file="stopwords.txt"):
    return [i.strip() for i in open(file,"r",encoding="utf-8").readlines()]

def softmax(x):
    if len(x.shape) == 1:
        x = x.reshape(1,-1)
    ex = np.exp(x)
    return ex / np.sum(ex,axis=1)

def sigmoid( x):
    if x > 6:
        return 1.0
    elif x < -6:
        return 0.0
    else:
        return 1 / (1 + np.exp(-x))

def build_dict(data_list):
    word_2_index,index_2_word,word_times = {},{},{}
    for content in data_list:
        for word in content:
            word_times[word] = word_times.get(word,0) + 1
            if word not in word_2_index:
                now_index =  len(word_2_index)
                word_2_index[word] = word_2_index.get(word,now_index)
                index_2_word[now_index] = word
    vec = []
    for word,times in word_times.items():
        word_index = word_2_index[word]
        for i in range(times):
            vec.append(word_index)
    return word_2_index,index_2_word,vec


def sampling(table, count):
    indices = np.random.randint(low=0, high=len(table), size=count)
    return [table[i] for i in indices]

def get_sample_tuple(now_word,other_words,neg_nums,word_2_index,table):
    result = []
    now_word_index = word_2_index[now_word]
    for other_word in other_words:
        other_word_index = word_2_index[other_word]
        result.append((now_word_index,other_word_index,1))
        samples = sampling(table,neg_nums)
        for s in samples:
            if s == now_word_index or s == other_word_index:
                continue
            result.append((now_word_index,s,0))
    return result

if __name__ == "__main__":
    # np.random.seed(7)

    data_math = pd.read_csv("数学原始数据.csv", encoding='gbk', header=None)[0].values
    data_math = clean_info(data_math)
    word_2_index, index_2_word, table = build_dict(data_math)

    word_size = len(word_2_index)
    feature_num = 50

    n_gram = 4
    negative_num = 5
    lr = 0.01
    epochs = 4

    w1 = np.random.normal(size=(word_size, feature_num))
    w2 = np.random.normal(size=(feature_num, word_size))

    for e in range(epochs):
        for content in tqdm(data_math):
            for index,word in enumerate(content):
                other_words = content[max(0, index - n_gram):index] + content[index + 1: index + 1 + n_gram]
                samples = get_sample_tuple(word,other_words,negative_num,word_2_index,table)

                for now_word_index,other_word_index,is_context in samples:
                    h = 1 * w1[now_word_index]
                    p = h @ w2[:,other_word_index]
                    predict = sigmoid(p)

                    G2 = predict - is_context
                    delta_w2 = h.T * G2

                    G1 = G2 *  w2[:,other_word_index].T
                    delta_w1 = 1 * G1

                    w2[:, other_word_index] -= lr * delta_w2
                    w1[now_word_index] -= lr * delta_w1

    pickle.dump([w1, word_2_index, index_2_word, w2], open("_负采样_.pkl", "wb"))
