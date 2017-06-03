import os
import random
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import pickle
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
f = open('lexcion.pickle', 'rb')
lex = pickle.load(f)
f.close()


def get_random_line(file, point):
    file.seek(point)
    file.readline()
    return file.readline()


# 从文件中随机选择n条记录
def get_n_random_line(file_name, n=150):
    lines = []
    file = open(file_name, encoding='latin-1')
    total_bytes = os.stat(file_name).st_size
    for i in range(n):
        random_point = random.randint(0, total_bytes)
        lines.append(get_random_line(file, random_point))
    file.close()
    return lines


def get_test_dataset(test_file):
    with open(test_file, encoding='latin-1') as f:
        test_x = []
        test_y = []
        lemmatizer = WordNetLemmatizer()
        for line in f:
            label = line.split(':%:%:%:')[0]
            tweet = line.split(':%:%:%:')[1]
            words = word_tokenize(tweet.lower())
            words = [lemmatizer.lemmatize(word) for word in words]
            features = np.zeros(len(lex))
            for word in words:
                if word in lex:
                    features[lex.index(word)] += 1

            test_x.append(list(features))
            test_y.append(eval(label))
    return test_x, test_y


test_x, test_y = get_test_dataset('testing.csv')

#######################################################################

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
n_output_layer = 3  # 输出层


def neural_network(data):
    layer_1_w_b = {'w_': tf.Variable(tf.random_normal([n_input_layer, n_layer_1]),name="Weight_1"),
                   'b_': tf.Variable(tf.random_normal([n_layer_1]),name="bias_1")}
    # 定义第二层"神经元"的权重和biases
    layer_2_w_b = {'w_': tf.Variable(tf.random_normal([n_layer_1, n_layer_2]),name="Weight_2"),
                   'b_': tf.Variable(tf.random_normal([n_layer_2]),name="bias_2")}

    layer_3_w_b = {'w_': tf.Variable(tf.random_normal([n_layer_2, n_layer_3]),name="Weight_3"),
                   'b_': tf.Variable(tf.random_normal([n_layer_3]),name="bias_3")}

    layer_4_w_b = {'w_': tf.Variable(tf.random_normal([n_layer_3, n_layer_4]),name="Weight_4"),
                   'b_': tf.Variable(tf.random_normal([n_layer_4]),name="bias_4")}

    layer_5_w_b = {'w_': tf.Variable(tf.random_normal([n_layer_4, n_layer_5]),name="Weight_5"),
                   'b_': tf.Variable(tf.random_normal([n_layer_5]),name="bias_5")}

    layer_6_w_b = {'w_': tf.Variable(tf.random_normal([n_layer_5, n_layer_6]),name="Weight_6"),
                   'b_': tf.Variable(tf.random_normal([n_layer_6]),name="bias_6")}

    layer_7_w_b = {'w_': tf.Variable(tf.random_normal([n_layer_6, n_layer_7]),name="Weight_7"),
                   'b_': tf.Variable(tf.random_normal([n_layer_7]),name="bias_7")}

    layer_8_w_b = {'w_': tf.Variable(tf.random_normal([n_layer_7, n_layer_8]),name="Weight_8"),
                   'b_': tf.Variable(tf.random_normal([n_layer_8]),name="bias_8")}

    layer_9_w_b = {'w_': tf.Variable(tf.random_normal([n_layer_8, n_layer_9]),name="Weight_9"),
                   'b_': tf.Variable(tf.random_normal([n_layer_9]),name="bias_9")}

    layer_10_w_b = {'w_': tf.Variable(tf.random_normal([n_layer_9, n_layer_10]),name="Weight_10"),
                    'b_': tf.Variable(tf.random_normal([n_layer_10]),name="bias_10")}

    # 定义输出层"神经元"的权重和biases
    layer_output_w_b = {'w_': tf.Variable(tf.random_normal([n_layer_10, n_output_layer]),name="Weight_Out"),
                        'b_': tf.Variable(tf.random_normal([n_output_layer]),name="bias_out")}

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
    '''
    sq_1 = tf.reduce_sum(tf.square(layer_1_w_b['w_']))
    sq_2 = tf.reduce_sum(tf.square(layer_2_w_b['w_']))
    sq_3 = tf.reduce_sum(tf.square(layer_3_w_b['w_']))
    sq_4 = tf.reduce_sum(tf.square(layer_4_w_b['w_']))
    sq_5 = tf.reduce_sum(tf.square(layer_5_w_b['w_']))
    sq_6 = tf.reduce_sum(tf.square(layer_6_w_b['w_']))
    sq_7 = tf.reduce_sum(tf.square(layer_7_w_b['w_']))
    sq_8 = tf.reduce_sum(tf.square(layer_8_w_b['w_']))
    sq_9 = tf.reduce_sum(tf.square(layer_9_w_b['w_']))
    sq_10 = tf.reduce_sum(tf.square(layer_10_w_b['w_']))

    a = tf.add(sq_1, sq_2)
    l = [sq_3, sq_4, sq_5, sq_6, sq_7, sq_8, sq_9, sq_10]
    for s in l:
        a = tf.add(a, s)
    '''
    # a=tf.add(tf.add(tf.reduce_sum(tf.square(layer_1_w_b['w_'])),tf.reduce_sum(tf.square(layer_2_w_b['w_']))),tf.reduce_sum(tf.square(layer_output_w_b['w_'])))
    return layer_output

X = tf.placeholder('float')
Y = tf.placeholder('float')
batch_size = 150

def train_neural_network(X, Y):
    save_path="C:/Users/Jz Chai/PycharmProjects/TFexercice/Ex2/tmp/model.ckpt"
    predict = neural_network(X)
    cost_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=Y))
    optimizer = tf.train.AdamOptimizer().minimize(cost_func)
    #print_tensors_in_checkpoint_file(file_name=save_path, tensor_name='Weight_Out',all_tensors=False)
    with tf.Session() as session:


        lemmatizer = WordNetLemmatizer()
        saver = tf.train.Saver()
        i = 0
        pre_accuracy = 0
        if os.path.exists("C:/Users/Jz Chai/PycharmProjects/TFexercice/Ex2/tmp/model.ckpt.index"):
            saver.restore(session, save_path)
            print("a")
        else:
            session.run(tf.global_variables_initializer())

        while True:  # 一直训练
            batch_x = []
            batch_y = []

            # if model.ckpt文件已存在:
            #	saver.restore(session, 'model.ckpt')  恢复保存的session


            try:
                lines = get_n_random_line('training.csv', batch_size)
                for line in lines:
                    label = line.split(':%:%:%:')[0]
                    tweet = line.split(':%:%:%:')[1]
                    words = word_tokenize(tweet.lower())
                    words = [lemmatizer.lemmatize(word) for word in words]

                    features = np.zeros(len(lex))
                    for word in words:
                        if word in lex:
                            features[lex.index(word)] += 1  # 一个句子中某个词可能出现两次,可以用+=1，其实区别不大

                    batch_x.append(list(features))
                    batch_y.append(eval(label))

                session.run([optimizer, cost_func], feed_dict={X: batch_x, Y: batch_y})
            except Exception as e:
                print(e)

            # 准确率
            if i > 0:
                correct = tf.equal(tf.argmax(predict, 1), tf.argmax(Y, 1))
                accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
                accuracy = accuracy.eval({X: test_x, Y: test_y})
                if accuracy > pre_accuracy:  # 保存准确率最高的训练模型
                    print('准确率: ', accuracy)
                    pre_accuracy = accuracy

                    s=saver.save(session,save_path) # 保存sessionsave_path
                    print(s)
                i = 0
            i += 1
            print (i)


#train_neural_network(X, Y)

def prediction(tweet_text):
    predict = neural_network(X)
    saver = tf.train.Saver()
    save_path = "C:/Users/Jz Chai/PycharmProjects/TFexercice/Ex2/tmp/model.ckpt"
    with tf.Session() as session:
        #session.run(tf.global_variables_initializer())


        print("b")
        #save_path = "C:/Users/Jz Chai/PycharmProjects/TFexercice/Ex2/model.ckpt"
        saver.restore(session, save_path)
        print ("c")

        lemmatizer = WordNetLemmatizer()
        words = word_tokenize(tweet_text.lower())
        words = [lemmatizer.lemmatize(word) for word in words]

        features = np.zeros(len(lex))
        for word in words:
            if word in lex:
                features[lex.index(word)] += 1

        # print(predict.eval(feed_dict={X:[features]})) [[val1,val2,val3]]
        res = session.run(tf.argmax(predict.eval(feed_dict={X: [features]}), 1))
        if res==0:
            a="positive"
        elif res==1:
            a="neutral"
        elif res==2:
            a="negative"
        session.close()

        return a

#print(prediction("I am very happy"))
#print(prediction("Sad like hell!"))

