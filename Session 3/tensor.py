import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

x1 = tf.constant(5)
x2 = tf.constant(6)

result = tf.multiply(x1, x2)
# print(result)

# sess = tf.Session()
# output = sess.run(result)
# print(output)
# sess.close()

with tf.Session() as sess:
    output = sess.run(result)
    print(output)