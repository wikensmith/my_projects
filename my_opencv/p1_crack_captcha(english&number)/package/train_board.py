import copy

import tensorflow as tf
import cv2 as cv
import numpy as np
from my_modules.utils import cur_time
from pic_binary import PicBinary


learning_rate = 0.0001
dropout = 0.8
max_steps = 3000

log_dir = './log'

sess = tf.InteractiveSession()
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
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2( x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def variable_summeries(var):
    print(var,"var in variable summeries")
    with tf.name_scope('summaries'):
        # mean = tf.reduce_mean('mean', var)
        # tf.summary.scalar('mean', mean)
        # with tf.name_scope('stddev'):
        #     stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        # tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


with tf.name_scope('input'):
    x = tf. placeholder(tf.float32, [100, 4800])
    y_ = tf.placeholder(tf.float32, [100, 62])  # 36*36+36   (所有可能结果的组合)


with tf.name_scope("img_reshape"):
    img_reshaped = tf.reshape(x, [-1, 60, 80, 1])
    tf.summary.image('input', img_reshaped)


def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    with tf.name_scope(layer_name):
        with tf.name_scope("weights"):
            weights = weight_variable([input_dim, output_dim])
            variable_summeries(weights)
        with tf.name_scope("biases"):
            biases = bias_variable([output_dim])
            variable_summeries(biases)
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights)+biases
            tf.summary.histogram('pre_activate', preactivate)
        activations = act(preactivate, name='activation')
        return activations


hidden1 = nn_layer(x, 4800, 1024, 'layer1')


with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    tf.summary.scalar('dropout_keep_probability', keep_prob)
    dropped = tf.nn.dropout(hidden1, keep_prob)

y = nn_layer(dropped, 1024, 62, 'layer2', act=tf.identity)

with tf.name_scope('cross_entropy'):
    diff = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_)
    with tf.name_scope('total'):
        cross_entropy = tf.reduce_mean(diff)
    tf.summary.scalar('cross_entropy', cross_entropy)


with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)

train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
test_writer = tf.summary.FileWriter(log_dir + "/test")

merged = tf.summary.merge_all()



def get_pic(batch):
    ac = PicBinary().main()
    n = 0
    y_real = np.zeros((1, 62))
    img = np.zeros((1, 4800))
    while True:
        text, image = ac.__next__()
        if not text:
            continue
        temp_y = np.zeros([62], np.uint8)
        if str(text) not in alph_lst or not isinstance(image, np.ndarray) or len(text)==2:
            continue
        temp_img = image.ravel()
        index = alph_lst.index(str(text))
        temp_y[index] = 1
        y_real = np.vstack((y_real, temp_y))
        img = np.vstack((img, temp_img))
        n += 1
        if n == (batch-1):
            break
    return img, y_real


def feed_dict(train):
    if train:
        xs, ys = get_pic(100)
        k = dropout
    else:
        xs, ys = get_pic(100)
        k = 1
    # print(ys.shape, "shape in feed", type(ys))
    return {x: xs, y_: ys, keep_prob: k}


init = tf.global_variables_initializer()
sess.run(init)
alph_lst = gen_dic()
saver = tf.train.Saver()
for i in range(max_steps):
    print(i, " 次 当前时间：", cur_time())
    if i % 5 == 0 and i != 0:
        summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
        # summary, acc = sess.run(accuracy, feed_dict=feed_dict(False))
        test_writer.add_summary(summary, i)
        print('Accuracy at step %s: %f' % (i, acc))
    else:
        if i % 100 == 0 and i != 0:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True),
            # summary, _ = sess.run(train_step, feed_dict=feed_dict(True),
                                  options=run_options, run_metadata=run_metadata)
            train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
            train_writer.add_summary(summary, i)
            saver.save(sess, log_dir+'model_board.ckpt', i)
            print("Adding run metadata for",i)
        else:
            summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
            # summary, _ = sess.run(train_step, feed_dict=feed_dict(True))
            train_writer.add_summary(summary, i)
train_writer.close()
test_writer.close()







