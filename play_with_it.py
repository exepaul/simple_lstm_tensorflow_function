import tensorflow as tf
import numpy as np
a=tf.placeholder(tf.float32,shape=[1,4,4])


with tf.variable_scope('encoder') as scope:
    cell=tf.nn.rnn_cell.LSTMCell(2)
    super12=tf.nn.bidirectional_dynamic_rnn(cell,cell,inputs=a,dtype=tf.float32)

with tf.Session() as tt:
    tt.run(tf.global_variables_initializer())
    ar=tt.run(super12,feed_dict={a:np.reshape(np.random.randint(0,10,[1,4,4]),[1,4,4])})
    new,(x,r)=ar
    print(new)
    print(new[0].shape)
