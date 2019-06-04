import copy

import tensorflow as tf
import cv2 as cv
import numpy as np

from pic_binary import PicBinary


def gen_dic():
    temp_lst = []
    for i in range(10):
        temp_lst.append(str(i))
    for i in range(65, 91):
        temp_lst.append(chr(i))
    for i in range(97, 123):
        temp_lst.append(chr(i))
    alph_lst = copy.deepcopy(temp_lst)
    for i in temp_lst:
        for j in temp_lst:
            alph_lst.append(str(i) + str(j))
    return alph_lst


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    print(initial, "initial")
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2( x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


x = tf. placeholder(tf.float32, [None, 4800])
y_ = tf.placeholder(tf.float32, [None, 3906])  # 36*36+36   (所有可能结果的组合)
img = tf.reshape(x, [-1, 60, 80, 1])
w_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(img, w_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

w_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

w_fc1 = weight_variable([15 * 20 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 15 * 20 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

w_fc2 = weight_variable([1024, 3906])
b_fc2 = bias_variable([3906])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv),
                                              reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


ac = PicBinary().main()
sess = tf.InteractiveSession()
saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=1)
tf.global_variables_initializer().run()
alph_lst = gen_dic()
for i in range(3000):
    print(f"第 {i} 次")
    n = 0
    y_real = np.zeros((1, 3906))
    img = np.zeros((1, 4800))
    while True:
        text, image = ac.__next__()
        if not text:
            continue
        temp_y = np.zeros([3906], np.uint8)
        # print(text, type(image))
        if str(text) not in alph_lst or not isinstance(image, np.ndarray):
            continue
        temp_img = image.ravel()
        index = alph_lst.index(str(text))
        temp_y[index] = 1
        y_real = np.vstack((y_real, temp_y))
        img = np.vstack((img, temp_img))
        n += 1
        if n == 49:
            break
    np.delete(y_, 0, axis=0)
    np.delete(img, 0, axis=0)
    # print(y_real.shape, type(y_real), "shape of y_")
    # print(img.shape,type(img),  "img shape")

    train_step.run(feed_dict={x: img, y_: y_real, keep_prob: 0.5})
    if i % 5 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: img, y_: y_real,
                                                  keep_prob: 1.0})
        print(f"进行{i}次训练, 准确率是{train_accuracy}%")
    saver.save(sess, './my_moduls/recognise_model')




