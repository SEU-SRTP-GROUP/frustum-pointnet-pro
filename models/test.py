import tensorflow as tf
import numpy as np
input = tf.TensorArray(size=2,dtype=tf.int32)
input=input.write(0,tf.constant(np.arange(24).reshape([8,3])))
input=input.write(1,tf.constant(np.arange(24,48).reshape([8,3])))
with tf.Session() as sess:
    print(sess.run(tf.constant(np.arange(24).reshape([8, 3]))))
    print(sess.run(tf.constant(np.arange(24,48).reshape([8, 3]))))
    a = input.stack()
    a = tf.transpose(a,[1,0,2])
    print(sess.run(a))
    print(a)

    # [batch,n,c]
    # [n,batch,c]
