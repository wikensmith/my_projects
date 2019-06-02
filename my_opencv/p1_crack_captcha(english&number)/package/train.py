"""
author: wiken
Date:2019/6/2
"""
import copy

import tensorflow as tf
import cv2 as cv
import numpy as np

from pic_binary import PicBinary


class Train:
    def __init__(self):
        self.alph_lst = []
        self.gen_dic()

    def gen_dic(self):
        temp_lst = []
        for i in range(10):
            temp_lst.append(i)
        for i in range(65, 91):
            temp_lst.append(chr(i))
        for i in range(97, 123):
            temp_lst.append(chr(i))
        self.alph_lst = copy.deepcopy(temp_lst)
        for i in temp_lst:
            for j in temp_lst:
                self.alph_lst.append(str(i) + str(j))

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, w):
        return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def net(self):
        self.x = tf. placeholder(tf.float32, [None, 4800])
        self.y_ = tf.placeholder(tf.float32, [None, 3096])  # 36*36+36   (所有可能结果的组合)
        img = tf.reshape(self.x, [-1, 60, 80, 1])
        w_conv1 = self.weight_variable([5, 5, 1, 32])
        b_conv1 = self.bias_variable([32])
        h_conv1 = tf.nn.relu(self.conv2d(img, w_conv1) + b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)

        w_conv2 = self.weight_variable([5, 5, 32, 64])
        b_conv2 = self.bias_variable([64])
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, w_conv2) + b_conv2)
        h_pool2 = self.max_pool_2x2(h_conv2)

        w_fc1 = self.weight_variable([15 * 20 * 64, 1024])
        b_fc1 = self.bias_variable([1024])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 15 * 20 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

        self.keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        w_fc2 = self.weight_variable([1024, 3096])
        b_fc2 = self.bias_variable([3096])
        y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(y_conv),
                                                      reduction_indices=[1]))
        train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(self.y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return train_step, accuracy

    def main(self):
        ac = PicBinary().main()
        sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        train_step, accuracy = self.net()
        for i in range(3000):
            if i % 20 == 0:
                print(f"第 {i} 次")
            n = 0
            y_ = np.zeros((1, 2))
            img = np.zeros((1, 4800))
            while True:
                text, image = ac.__next__()
                if not text:
                    continue
                temp_y = np.zeros([3906], np.uint8)
                if text not in self.alph_lst and isinstance(image, np.ndarray):
                    continue
                temp_img = image.ravel()
                index = self.alph_lst.index(text)
                temp_y[index] = 1
                np.vstack((y_, temp_y))
                np.vstack((img, temp_img))
                n += 1
                if n == 50:
                    break

            train_step.run(feed_dict={self.x:img, self.y_:y_, self.keep_prob:0.5})
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={self.x: img, self.y_: y_,
                                                          self.keep_prob: 1.0})
                print(f"进行{i}次训练, 准确率是{train_accuracy}%")


if __name__ == '__main__':
    Train().main()


