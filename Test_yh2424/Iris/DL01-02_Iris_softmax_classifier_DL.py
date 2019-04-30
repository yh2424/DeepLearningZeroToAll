# Lab 6 Softmax Classifier
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.set_random_seed(777)  # for reproducibility

# Predicting animal type based on various features
xy = np.loadtxt('Test_yh2424/Iris/data-01_iris_edited.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

print(x_data.shape, y_data.shape)

nb_classes = 3 # 0 ~ 2

X = tf.placeholder(tf.float32, [None, 4])
Y = tf.placeholder(tf.int32, [None, 1])  # 0 ~ 2

Y_one_hot = tf.one_hot(Y, nb_classes)  # one hot
print("one_hot:", Y_one_hot)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])
print("reshape one_hot:", Y_one_hot)

'''
one_hot: Tensor("one_hot:0", shape=(?, 1, 7), dtype=float32)
reshape one_hot: Tensor("Reshape:0", shape=(?, 7), dtype=float32)
'''

W1 = tf.Variable(tf.random_normal([4, nb_classes]), name='weight1')
L1 = tf.matmul(X, W1)
# L1= tf.nn.relu(tf.matmul(X, W1))

W2 = tf.Variable(tf.random_normal([3, nb_classes]), name='weight2')
logits = tf.matmul(L1, W2)

hypothesis = tf.nn.softmax(logits)

# Cross entropy cost/loss
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,
                                                                 labels=tf.stop_gradient([Y_one_hot])))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# List parameters for graph
cost_gr = []
accuracy_gr = []
Epoch_gr = []

# Launch results
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        _, cost_val, acc_val = sess.run([optimizer, cost, accuracy], feed_dict={X: x_data, Y: y_data})

        cost_gr.append(cost_val)
        accuracy_gr.append(acc_val)
        Epoch_gr.append(step)

        if step % 10 == 0:
            print("Step: {:5}\tCost: {:.3f}\tAcc: {:.2%}".format(step, cost_val, acc_val))


    # # Let's see if we can predict
    # pred = sess.run(prediction, feed_dict={X: x_data})
    # # y_data: (N,1) = flatten => (N, ) matches pred.shape
    # for p, y in zip(pred, y_data.flatten()):
    #     print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))



plt.plot(Epoch_gr, accuracy_gr)



