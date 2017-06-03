import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import pickle
import numpy as np
import pandas as pd
import dask.dataframe as dd
from collections import OrderedDict
import csv
import codecs

org_train_file = 'training.1600000.processed.noemoticon.csv'
org_test_file = 'testdata.manual.2009.06.14.csv'



# 提取文件中有用的字段
"""
def useful_f(org_file, output_file):
    output = open(output_file, 'w')

    f=csv.reader(codecs.open(org_file, 'rU'))
    try:
        for line in f:  # "4","2193601966","Tue Jun 16 08:40:49 PDT 2009","NO_QUERY","AmandaMarie1028","Just woke up. Having no school is the best feeling ever "
            # line = line.replace('"', '')
            #line = line.split('"')
            tweet = line[-1]
            clf = line[0]

            if clf == '0':
                clf = [0, 0, 1]  # 消极评论
            elif clf == '2':
                clf = [0, 1, 0]  # 中性评论
            elif clf == '4':
                clf = [1, 0, 0]  # 积极评论

            outputline = str(clf) + ':%:%:%:' + tweet+'\n'

            #writer.writerow([outputline])
            output.write(outputline)
            #output.write(outputline)  # [0, 0, 1]:%:%:%: that's a bummer.  You shoulda got David Carr of Third Day to do it. ;D

    except Exception as e:
        print(e)

    output.close()  # 处理完成，处理后文件大小127.5M

useful_f(org_train_file, 'training.csv')
useful_f(org_test_file, 'testing.csv')


# 创建词汇表
def create_lexicon(train_file):
    lex = []
    lemmatizer = WordNetLemmatizer()
    with open(train_file, buffering=10000, encoding='latin-1') as f:
        try:
            count_word = {}  # 统计单词出现次数
            for line in f:
                tweet = line.split(':%:%:%:')[1]
                words = word_tokenize(tweet.lower())
                for word in words:
                    word = lemmatizer.lemmatize(word)
                    if word not in count_word:
                        count_word[word] = 1
                    else:
                        count_word[word] += 1

            count_word = OrderedDict(sorted(count_word.items(), key=lambda t: t[1]))
            common_word=["haha","!","!!","sad","angry","ass","fuck","happy","annoying","angry","frustrated","kill","die","XD","=(","damn"]
            discard=["this","that","then","than","the","those","these","he","she"]
            for word in count_word:
                if count_word[word] < 100000 and count_word[word] >100 and len(word)>2 and ("'" not in word) and (word not in discard) or (word in common_word):  # 过滤掉一些词
                    lex.append(word)
        except Exception as e:
            print(e)
    return lex


lex = create_lexicon('training.csv')

with open('lexcion.pickle', 'wb') as f:
    pickle.dump(lex, f)
"""
'''

# 把字符串转为向量
def string_to_vector(input_file, output_file, lex):
    output_f = open(output_file, 'w')
    lemmatizer = WordNetLemmatizer()
    with open(input_file, buffering=10000, encoding='latin-1') as f:
        for line in f:
            line = line.replace(' ', '')
            label = line.split(':%:%:%:')[0]
            tweet = line.split(':%:%:%:')[1]
            words = word_tokenize(tweet.lower())
            words = [lemmatizer.lemmatize(word) for word in words]

            features = np.zeros(len(lex))
            for word in words:
                if word in lex:
                    features[lex.index(word)] += 1  # 一个句子中某个词可能出现两次,可以用+=1，其实区别不大

            features = list(features)
            output_f.write(str(label) + ":" + str(features) + '\n')
    output_f.close()

f = open('lexcion.pickle', 'rb')
lex = pickle.load(f)
f.close()

# lexcion词汇表大小112k,training.vec大约112k*1600000  170G  太大，只能边转边训练了
#string_to_vector('training.csv', 'training.vec', lex)
#string_to_vector('testing.csv', 'testing.vec', lex)
'''
'''
f = open('testing.vec', 'r')
i=0
for line in f:

    if i==0:
        v=eval(line.split(':')[1])
        print(eval(line.split(':')[0]))
        print(v[12])
        i+=1
f.close()
'''

