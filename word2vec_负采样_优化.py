import pandas as pd
import jieba
import numpy as np
import re
from tqdm import tqdm
import pickle


def get_data(file = "new_data.txt"):
    with open(file,encoding='utf-8') as f:
        datas = f.read().split("\n")
    word_2_index = {}
    index_2_word = []

    s_table = []
    for sentence in datas:
        words = sentence.split(" ")
        for w in words:
            if w not in word_2_index:
                word_2_index[w] = len(word_2_index)
                index_2_word.append(w)
            s_table.append(word_2_index[w])

    return datas , word_2_index,index_2_word,s_table


def softmax(x):
    max_x = np.sum(x,axis = 1,keepdims = True)
    ex = np.exp(x)
    result = ex / np.sum(ex,axis = 1,keepdims = True)
    return np.clip(result,1e-20,1)

def sigmoid(x):
    x = np.clip(x,-50,50)
    return 1/(1+np.exp(-x))


def make_samples(sentence,index):

    global word_2_index,corpus_len, neg_rate,s_table

    now_word_index = word_2_index[sentence[index]] # [ ]

    other_words = sentence[max(0, index - n_gram): index] + sentence[index + 1: index + n_gram + 1]
    other_words_index = [word_2_index[i] for i in other_words]

    # all_neg_index = [i for i in range(corpus_len) if i not in other_words_index + [now_word_index]]
    # s_table
    t = np.random.randint(0,len(s_table),size = (neg_rate*len(other_words_index)))
    t = [i for i in t if i not in other_words_index + [now_word_index]]


    samples = [ ]

    for i in other_words_index:
        samples.append((now_word_index,i,1))

    for i in t:
        samples.append((now_word_index,s_table[i],0))

    return pro_samples(samples)

def pro_samples(samples):

    # now_word_index = []
    other_word_index = []
    label = []
    for sample in samples:
        # now_word_index.append(sample[0])
        other_word_index.append(sample[1])
        label.append(sample[2])
    return samples[0][0],other_word_index,np.array(label).reshape(1,-1)

if __name__ == "__main__":
    all_datas, word_2_index,index_2_word,s_table = get_data()

    corpus_len = len(word_2_index)
    embedding_num = 50
    epoch = 4
    lr = 0.01
    n_gram = 3
    neg_rate = 5

    w1 = np.random.normal(0,1,size = (corpus_len,embedding_num))

    w2 = np.random.normal(0,1,size = w1.T.shape)

    # skip_gram
    for e in range(epoch):
        for sentence in tqdm(all_datas):
            sentence =sentence.split(" ")
            for now_idx_sent,now_word in enumerate(sentence):

                now_word_index,other_word_index,label = make_samples(sentence,now_idx_sent)

                # for now_word_index,other_word_index,label in samples:
                hidden = 1 *  w1[None,now_word_index]
                pre = hidden @ w2[:,other_word_index]

                pro = sigmoid(pre)

                # loss = -np.sum(label * np.log(pro) + (1-label) * np.log(1-pro))

                G2 = pro - label
                delta_w2 = hidden.T @ G2

                G1 = G2 @ w2[:,other_word_index].T
                delta_w1 = G1

                w1[None,now_word_index] -= lr * delta_w1
                w2[:,other_word_index] -= lr * delta_w2 / len(label)

    with open("vec.pkl2","wb") as f:
        pickle.dump([w1,w2,word_2_index,index_2_word],f)



