import tensorflow as tf
import numpy as np

a=tf.placeholder(tf.float32,shape=[2,4,4])
data=np.reshape(np.random.randint(0,10,[2,4,4]),[2,4,4])
print(data)

with tf.variable_scope('encoder') as scope:
    cell=tf.nn.rnn_cell.LSTMCell(4)
    outputs, (ffs, fbs) =tf.nn.bidirectional_dynamic_rnn(cell,cell,inputs=a,dtype=tf.float32)
    outputs_f, outputs_b = outputs
    print(outputs_f, outputs_b) # batch-major form
    outputs_f_tm = tf.transpose(outputs_f, [1, 0, 2])
    outputs_b_tm = tf.transpose(outputs_b, [1, 0, 2])
    print(outputs_f_tm, outputs_b_tm) # time-major form
    print(outputs_f_tm[-1], outputs_b_tm[-1])


with tf.Session() as tt:
    tt.run(tf.global_variables_initializer())
    outputs_,first,second=tt.run([outputs,ffs,fbs],feed_dict={a:data})

    print(outputs_)

    print("output yea",outputs_)

    print('\n\n\n\n\n\n')

    print("first",first)
