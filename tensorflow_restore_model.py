# coding=utf-8
import tensorflow as tf
import numpy as np


money = np.array([[109], [82], [99], [72], [87], [78], [86], [84], [94], [57]]).astype(np.float32)
click = np.array([[11], [8], [8], [6], [7], [7], [7], [8], [9], [5]]).astype(np.float32)
x_test = money[0:5].reshape(-1, 1)
y_test = click[0:5]

with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('./tf_test_model/tf_test-100.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('./tf_test_model/'))
    graph = tf.get_default_graph()
    x = graph.get_operation_by_name('x').outputs[0]
    y = tf.get_collection("pred_network")[0]
    print("测试集的预测值是:\n", sess.run(y, feed_dict={x: [[200]]}))
