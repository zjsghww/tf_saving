# coding=utf-8
import tensorflow as tf
import numpy as np

money = np.array([[109], [82], [99], [72], [87], [78], [86], [84], [94], [57]]).astype(np.float32)
click = np.array([[11], [8], [8], [6], [7], [7], [7], [8], [9], [5]]).astype(np.float32)
x_test = money[0:5].reshape(-1, 1)
y_test = click[0:5]
x_train = money[5:].reshape(-1, 1)
y_train = click[5:]
# 模型的输入用占位符
x = tf.placeholder(tf.float32, [None, 1], name='x')
w = tf.Variable(tf.zeros([1, 1]))
b = tf.Variable(tf.zeros([1]))
# 模型的预测值
y = tf.matmul(x, w) + b
tf.add_to_collection('pred_network', y)
# 模型的真实输出值用占位符
y_ = tf.placeholder(tf.float32, [None, 1])
# 计算真实值与预测值之间的差距，并用梯度下降优化
cost = tf.reduce_sum(tf.pow((y - y_), 2))
train_step = tf.train.GradientDescentOptimizer(0.000001).minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
cost_history = []

for i in range(30):
    # 将训练数据喂给模型的输入
    feed = {x: x_train, y_: y_train}
    sess.run(train_step, feed_dict=feed)
    cost_history.append(sess.run(cost, feed_dict=feed))

# 输出最终的W,b和cost值
print("测试集合的预测值:\n", sess.run(y, feed_dict={x: x_test}))
print("W_Value: %f" % sess.run(w), "b_Value: %f" % sess.run(b), "cost_Value: %f" % sess.run(cost, feed_dict=feed))
print("cost history: ", cost_history)

saver = tf.train.Saver()
saver_path = saver.save(sess, "./tf_test_model/tf_test", global_step=100)
print("model saved in file: ", saver_path)

