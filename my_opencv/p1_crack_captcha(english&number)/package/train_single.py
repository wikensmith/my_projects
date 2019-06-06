"""
author: wiken
Date:2019/6/3
"""
import copy

import tensorflow as tf
import cv2 as cv
import numpy as np
from pic_binary import PicBinary

sess = tf.InteractiveSession()


def gen_dic():
    temp_lst = []
    for i in range(10):
        temp_lst.append(i)
    for i in range(65, 91):
        temp_lst.append(chr(i))
    for i in range(97, 123):
        temp_lst.append(chr(i))
    alph_lst = copy.deepcopy(temp_lst)
    for i in temp_lst:
        for j in temp_lst:
            alph_lst.append(str(i) + str(j))
    return alph_lst


def weight_variable(shape, name=None):
    with tf.name_scope("weights"):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name=name)


def bias_variable(shape, name=None):
    with tf.name_scope("biases"):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name=name)


def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x, name=None):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


x = tf. placeholder(tf.float32, [None, 4800], name="x")
y_ = tf.placeholder(tf.float32, [None, 62], name="y")  # 36*36+36   (所有可能结果的组合)
img = tf.reshape(x, [-1, 60, 80, 1], name='images')
with tf.name_scope(name='conv1'):
    w_conv1 = weight_variable([5, 5, 1, 32], name="w")
    b_conv1 = bias_variable([32], name='b')
    h_conv1 = tf.nn.relu(conv2d(img, w_conv1) + b_conv1, name='ac')
    h_pool1 = max_pool_2x2(h_conv1, name='pool1')
    tf.summary.histogram("weights", w_conv1)
    tf.summary.histogram('biases', b_conv1)
    tf.summary.histogram('pool', h_pool1)

with tf.name_scope(name='conv2'):
    w_conv2 = weight_variable([5, 5, 32, 64], name='w')
    b_conv2 = bias_variable([64], name='b')
    h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2, name='conv2')
    h_pool2 = max_pool_2x2(h_conv2, name='pool2')
    tf.summary.histogram("weights", w_conv2)
    tf.summary.histogram('biases', b_conv2)
    tf.summary.histogram('pool', h_pool2)

with tf.name_scope(name="fc1"):
    w_fc1 = weight_variable([15 * 20 * 64, 1024], name='fc_w')
    b_fc1 = bias_variable([1024],name='fc_b')
    h_pool2_flat = tf.reshape(h_pool2, [-1, 15 * 20 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1, name='fc1_ac')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob, name='fc1_dropout')
    tf.summary.histogram("fc_w", w_fc1)
    tf.summary.histogram('fc_b', b_fc1)
    tf.summary.histogram('fc_ac', h_fc1)

with tf.name_scope(name='fc2'):
    w_fc2 = weight_variable([1024, 62], name='fc2_w')
    b_fc2 = bias_variable([62], name='fc2_b')
    y_fc2 = tf.matmul(h_fc1_drop, w_fc2) + b_fc2
    y_conv = tf.nn.softmax(y_fc2, name='y_conv')
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv),
                                                  reduction_indices=[1]), name='cross_entropy')
    tf.summary.histogram("fc_w", w_fc1)
    tf.summary.histogram('fc_b', b_fc1)
    tf.summary.histogram('fc_ac', h_fc1)

with tf.name_scope(name='train'):
    train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy, name='train')
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('./log/train_single')
writer.add_graph(sess.graph)


ac = PicBinary().main()
alph_lst = gen_dic()

saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=1)
tf.global_variables_initializer().run()
for i in range(3000):
    print(f"第 {i} 次")
    n = 0
    y_real = np.zeros((1, 62))
    img = np.zeros((1, 4800))
    while True:
        text, image = ac.__next__()
        if not text or len(text) == 2:
            continue
        temp_y = np.zeros([62], np.uint8)
        if text not in alph_lst or not isinstance(image, np.ndarray):
            continue
        temp_img = image.ravel()
        index = alph_lst.index(text)
        temp_y[index] = 1
        y_real = np.vstack((y_real, temp_y))
        img = np.vstack((img, temp_img))
        n += 1
        if n == 49:
            break
    np.delete(y_real, 0, axis=0)
    np.delete(img, 0, axis=0)
    # print(y_real.shape, type(y_real), "shape of y_")
    # print(img.shape, type(img),  "img shape")
    if i % 10 == 0:
        s = sess.run(merged, feed_dict={x: img, y_: y_real, keep_prob: 0.8})

        # train_step.run(feed_dict={x: img, y_: y_real, keep_prob: 0.8})

    else:
        sess.run(train_step, feed_dict={x: img, y_: y_real, keep_prob: 0.8})
        # train_step.run(feed_dict={x: img, y_: y_real, keep_prob: 0.8})
    if i % 5 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: img, y_: y_real,
                                                  keep_prob: 1.0})
        print(f"进行{i}次训练, 准确率是{train_accuracy*100}%")
    saver.save(sess, './my_moduls/single_model')

