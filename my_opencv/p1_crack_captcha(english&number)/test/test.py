
import tensorflow as tf


d1 = [[1,1,1], [2,2,2]]
b2 = [[3],[3],[3]]
c = tf.matmul(d1, b2)
print(c)