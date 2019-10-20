import tensorflow as tf

input = tf.TensorArray(size=2,dtype=tf.int32)
input=input.write(0,tf.constant([1,2,3]))
input=input.write(1,tf.constant([4,5,6]))
with tf.Session() as sess:
    a = input.stack()
    print(sess.run(a))
