"""
Simple linear regression example in TensorFlow
This program tries to predict the number of thefts from 
the number of fire in the city of Chicago
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import xlrd

DATA_FILE = 'data/fire_theft.xls'

def huber_loss(labels,predictions,delta=1.0):
    residual=tf.abs(predictions-labels)
    condition=tf.less(residual,delta)
    small_res=0.5*tf.square(residual)
    large_res=delta*residual-0.5*tf.square(delta)
    return tf.where(condition,small_res,large_res)

# Phase 1: Assemble the graph
# Step 1: read in data from the .xls file
book = xlrd.open_workbook(DATA_FILE, encoding_override='utf-8')
sheet = book.sheet_by_index(0)
data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
n_samples = sheet.nrows - 1

# Step 2: create placeholders for input X (number of fire) and label Y (number of theft)
X=tf.placeholder(tf.float32,name="input")
Y=tf.placeholder(tf.float32,name="labelY")

# Step 3: create weight and bias, initialized to 0
# name your variables w and b
w=tf.Variable(0.04,name="weights_1",dtype=tf.float32)

u=tf.Variable(0.03,name="weights_2",dtype=tf.float32)

t=tf.Variable(0.05,name="weights_3",dtype=tf.float32)

z=tf.Variable(0.04,name="weights_4",dtype=tf.float32)

b=tf.Variable(0.0,name="bias",dtype=tf.float32)

# Step 4: predict Y (number of theft) from the number of fire
# name your variable Y_predicted
Y_predicted = X*X*X*X*z + X*X*X*t + X * u + b

# Step 5: use the square error as the loss function
# name your variable loss<
loss=tf.square(Y-Y_predicted,name='loss')
hl=huber_loss(Y,Y_predicted)

# Step 6: using gradient descent with learning rate of 0.01 to minimize loss
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

# Phase 2: Train our model
with tf.Session() as sess:
    # Step 7: initialize the necessary variables, in this case, w and b
    # TO - DO
    writer=tf.summary.FileWriter('./graphs',sess.graph)
    sess.run(tf.global_variables_initializer())
    # Step 8: train the model
    for i in range(100):  # run 100 epochs
        total_loss = 0
        for x, y in data:
            # Session runs optimizer to minimize loss and fetch the value of loss
            # TO DO: write sess.run()
            _,l=sess.run([optimizer,loss],feed_dict={X:x,Y:y})
            total_loss += l

        print("Epoch {0}: {1}".format(i, total_loss / n_samples))
    z_value,t_value,u_value,w_value,b_value=sess.run([z,t,u,w,b])
    print("%f %f %f %f %f"%(z_value,t_value,u_value,w_value,b_value))
    # plot the results
    X, Y = data.T[0], data.T[1]
    writer.close()
    #X, Y = data[:,0], data[:,1]
    plt.plot(X, Y, 'bo', label='Real data')
    plt.plot(X,  X*X*X*X*z_value + X*X*X*t_value + X * u_value + b_value, 'ro', label='Predicted data')
    plt.legend()
    plt.show()

