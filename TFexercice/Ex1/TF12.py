# python3

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import random
import pickle
from collections import Counter
import time
import nltk
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
"""
'I'm super man'
tokenize:
['I', ''m', 'super','man' ] 
"""
from nltk.stem import WordNetLemmatizer

"""
词形还原(lemmatizer)，即把一个任何形式的英语单词还原到一般形式，与词根还原不同(stemmer)，后者是抽取一个单词的词根。
"""

pos_file = 'pos.txt'
neg_file = 'neg.txt'


# 创建词汇表
def create_lexicon(pos_file, neg_file):
    lex = []

    # 读取文件
    def process_file(f):
        with open(f, 'r') as t:
            lex = []
            lines = t.readlines()
            # print(lines)
            for line in lines:
                words = word_tokenize(line.lower())
                lex += words
            return lex

    lex += process_file(pos_file)
    lex += process_file(neg_file)
    # print(len(lex))
    lemmatizer = WordNetLemmatizer()
    lex = [lemmatizer.lemmatize(word) for word in lex]  # 词形还原 (cats->cat)

    word_count = Counter(lex)
    # print(word_count)
    # {'.': 13944, ',': 10536, 'the': 10120, 'a': 9444, 'and': 7108, 'of': 6624, 'it': 4748, 'to': 3940......}
    # 去掉一些常用词,像the,a and等等，和一些不常用词; 这些词对判断一个评论是正面还是负面没有做任何贡献
    lex = []
    for word in word_count:
        if word_count[word] < 2000 and word_count[word] > 20:  # 这写死了，好像能用百分比
            lex.append(word)  # 齐普夫定律-使用Python验证文本的Zipf分布 http://blog.topspeedsnail.com/archives/9546
    return lex


lex = create_lexicon(pos_file, neg_file)


# lex里保存了文本中出现过的单词。

# 把每条评论转换为向量, 转换原理：
# 假设lex为['woman', 'great', 'feel', 'actually', 'looking', 'latest', 'seen', 'is'] 当然实际上要大的多
# 评论'i think this movie is great' 转换为 [0,1,0,0,0,0,0,1], 把评论中出现的字在lex中标记，出现过的标记为1，其余标记为0
def normalize_dataset(lex):
    dataset = []

    # lex:词汇表；review:评论；clf:评论对应的分类，[0,1]代表负面评论 [1,0]代表正面评论
    def string_to_vector(lex, review, clf):
        words = word_tokenize(review.lower())
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]

        features = np.zeros(len(lex))
        for word in words:
            if word in lex:
                features[lex.index(word)] = 1  # 一个句子中某个词可能出现两次,可以用+=1，其实区别不大
        return [features, clf]

    with open(pos_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            one_sample = string_to_vector(lex, line, [1, 0])  # [array([ 0.,  1.,  0., ...,  0.,  0.,  0.]), [1,0]]
            dataset.append(one_sample)
    with open(neg_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            one_sample = string_to_vector(lex, line, [0, 1])  # [array([ 0.,  0.,  0., ...,  0.,  0.,  0.]), [0,1]]]
            dataset.append(one_sample)

    # print(len(dataset))
    return dataset


dataset = normalize_dataset(lex)
random.shuffle(dataset)
"""
#把整理好的数据保存到文件，方便使用。到此完成了数据的整理工作
with open('save.pickle', 'wb') as f:
	pickle.dump(dataset, f)
"""

# 取样本中的10%做为测试数据
test_size = int(len(dataset) * 0.1)
dataset = np.array(dataset)

train_dataset = dataset[:-(test_size)]
test_dataset = dataset[-(test_size):]

# Feed-Forward Neural Network
# 定义每个层有多少'神经元''
n_input_layer = len(lex)  # 输入层

n_layer_1 = 2000 # hide layer
n_layer_2 = 1500  # hide layer(隐藏层)听着很神秘，其实就是除输入输出层外的中间层
n_layer_3 = 1000
n_layer_4 = 800
n_layer_5 = 600
n_layer_6 = 400
n_layer_7 = 200
n_layer_8 = 100
n_layer_9 = 50
n_layer_10 = 25

n_output_layer = 2  # 输出层


# 定义待训练的神经网络
def neural_network(data):
    # 定义第一层"神经元"的权重和biases
    layer_1_w_b = {'w_': tf.Variable(tf.random_normal([n_input_layer, n_layer_1])),
                   'b_': tf.Variable(tf.random_normal([n_layer_1]))}
    # 定义第二层"神经元"的权重和biases
    layer_2_w_b = {'w_': tf.Variable(tf.random_normal([n_layer_1, n_layer_2])),
                   'b_': tf.Variable(tf.random_normal([n_layer_2]))}

    layer_3_w_b = {'w_': tf.Variable(tf.random_normal([n_layer_2, n_layer_3])),
                   'b_': tf.Variable(tf.random_normal([n_layer_3]))}

    layer_4_w_b = {'w_': tf.Variable(tf.random_normal([n_layer_3, n_layer_4])),
                   'b_': tf.Variable(tf.random_normal([n_layer_4]))}

    layer_5_w_b = {'w_': tf.Variable(tf.random_normal([n_layer_4, n_layer_5])),
                   'b_': tf.Variable(tf.random_normal([n_layer_5]))}

    layer_6_w_b = {'w_': tf.Variable(tf.random_normal([n_layer_5, n_layer_6])),
                   'b_': tf.Variable(tf.random_normal([n_layer_6]))}

    layer_7_w_b = {'w_': tf.Variable(tf.random_normal([n_layer_6, n_layer_7])),
                   'b_': tf.Variable(tf.random_normal([n_layer_7]))}

    layer_8_w_b = {'w_': tf.Variable(tf.random_normal([n_layer_7, n_layer_8])),
                   'b_': tf.Variable(tf.random_normal([n_layer_8]))}

    layer_9_w_b = {'w_': tf.Variable(tf.random_normal([n_layer_8, n_layer_9])),
                   'b_': tf.Variable(tf.random_normal([n_layer_9]))}

    layer_10_w_b = {'w_': tf.Variable(tf.random_normal([n_layer_9, n_layer_10])),
                   'b_': tf.Variable(tf.random_normal([n_layer_10]))}

    # 定义输出层"神经元"的权重和biases
    layer_output_w_b = {'w_': tf.Variable(tf.random_normal([n_layer_10, n_output_layer])),
                        'b_': tf.Variable(tf.random_normal([n_output_layer]))}

    # w·x+b
    layer_1 = tf.add(tf.matmul(data, layer_1_w_b['w_']), layer_1_w_b['b_'])
    layer_1 = tf.nn.relu(layer_1)  # 激活函数

    layer_2 = tf.add(tf.matmul(layer_1, layer_2_w_b['w_']), layer_2_w_b['b_'])
    layer_2 = tf.nn.relu(layer_2)  # 激活函数

    layer_3 = tf.add(tf.matmul(layer_2, layer_3_w_b['w_']), layer_3_w_b['b_'])
    layer_3 = tf.nn.relu(layer_3)  # 激活函数

    layer_4 = tf.add(tf.matmul(layer_3, layer_4_w_b['w_']), layer_4_w_b['b_'])
    layer_4 = tf.nn.relu(layer_4)  # 激活函数

    layer_5 = tf.add(tf.matmul(layer_4, layer_5_w_b['w_']), layer_5_w_b['b_'])
    layer_5 = tf.nn.relu(layer_5)  # 激活函数

    layer_6 = tf.add(tf.matmul(layer_5, layer_6_w_b['w_']), layer_6_w_b['b_'])
    layer_6 = tf.nn.relu(layer_6)  # 激活函数

    layer_7 = tf.add(tf.matmul(layer_6, layer_7_w_b['w_']), layer_7_w_b['b_'])
    layer_7 = tf.nn.relu(layer_7)  # 激活函数

    layer_8 = tf.add(tf.matmul(layer_7, layer_8_w_b['w_']), layer_8_w_b['b_'])
    layer_8 = tf.nn.relu(layer_8)  # 激活函数

    layer_9 = tf.add(tf.matmul(layer_8, layer_9_w_b['w_']), layer_9_w_b['b_'])
    layer_9 = tf.nn.relu(layer_9)  # 激活函数

    layer_10 = tf.add(tf.matmul(layer_9, layer_10_w_b['w_']), layer_10_w_b['b_'])
    layer_10 = tf.nn.relu(layer_10)  # 激活函数

    layer_output = tf.add(tf.matmul(layer_10, layer_output_w_b['w_']), layer_output_w_b['b_'])


    sq_1=tf.reduce_sum(tf.square(layer_1_w_b['w_']))
    sq_2 = tf.reduce_sum(tf.square(layer_2_w_b['w_']))
    sq_3 = tf.reduce_sum(tf.square(layer_3_w_b['w_']))
    sq_4 = tf.reduce_sum(tf.square(layer_4_w_b['w_']))
    sq_5 = tf.reduce_sum(tf.square(layer_5_w_b['w_']))
    sq_6 = tf.reduce_sum(tf.square(layer_6_w_b['w_']))
    sq_7 = tf.reduce_sum(tf.square(layer_7_w_b['w_']))
    sq_8 = tf.reduce_sum(tf.square(layer_8_w_b['w_']))
    sq_9 = tf.reduce_sum(tf.square(layer_9_w_b['w_']))
    sq_10 = tf.reduce_sum(tf.square(layer_10_w_b['w_']))

    a=tf.add(sq_1,sq_2)
    l=[sq_3,sq_4,sq_5,sq_6,sq_7,sq_8,sq_9,sq_10]
    for s in l:
        a = tf.add(a, s)

    #a=tf.add(tf.add(tf.reduce_sum(tf.square(layer_1_w_b['w_'])),tf.reduce_sum(tf.square(layer_2_w_b['w_']))),tf.reduce_sum(tf.square(layer_output_w_b['w_'])))
    return layer_output,a


# 每次使用50条数据进行训练
batch_size = 50

X = tf.placeholder('float', [None, len(train_dataset[0][0])])
# [None, len(train_x)]代表数据数据的高和宽（矩阵），好处是如果数据不符合宽高，tensorflow会报错，不指定也可以。
Y = tf.placeholder('float')


# 使用数据训练神经网络
def train_neural_network(X, Y):
    predict,a = neural_network(X)
    cost_func = tf.add(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=Y)),tf.multiply(0.1,a))#tf.multiply(0,a)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.005).minimize(cost_func)  # learning rate 默认 0.001

    epochs = 13
    with tf.Session() as session:
        #writer = tf.summary.FileWriter('./graphs', session.graph)
        session.run(tf.global_variables_initializer())

        random.shuffle(train_dataset)
        train_x = train_dataset[:, 0]
        train_y = train_dataset[:, 1]
        test_x = test_dataset[:, 0]
        test_y = test_dataset[:, 1]
        for epoch in range(epochs):
            i=0
            epoch_loss=0

            while i < len(train_x):

                start = i
                end = i + batch_size

                batch_x = train_x[start:end]
                batch_y = train_y[start:end]

                _, c = session.run([optimizer, cost_func], feed_dict={X: list(batch_x), Y: list(batch_y)})
                epoch_loss += c
                i += batch_size

            #plt.plot( epoch, epoch_loss, 'bo')

            print(epoch, ' : ', epoch_loss)

            correct = tf.equal(tf.argmax(predict, 1), tf.argmax(Y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
            acc_train=accuracy.eval(feed_dict={X: list(train_x), Y: list(train_y)})
            acc_test=accuracy.eval(feed_dict={X: list(test_x), Y: list(test_y)})
            print('准确率 train: ', acc_train)
            print('准确率: ', acc_test)
            plt.plot(epoch, acc_train, 'g^',epoch,acc_test,'bo')
        #writer.close()
        plt.show()
#a=time.time()
train_neural_network(X, Y)
#b=time.time()
#print ("%f seconds"%(b-a))