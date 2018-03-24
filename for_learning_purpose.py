import tensorflow as tf
import numpy as np
_placeholder_value=np.random.randint(0, 2, [2, 4])
print(_placeholder_value)
print(_placeholder_value.shape)
class BasicClassifier():
    def __init__(self, wdim, hdim, vocab_size, num_labels):
        tf.reset_default_graph()

        # define placeholders
        name = tf.placeholder(tf.int32, [None, None], name='name')
        labels = tf.placeholder(tf.int32, [None, ], name='label')

        # expose placeholders
        self.placeholders = {'name': name, 'label': labels}

        # infer dimensions of batch
        batch_size_, seq_len_ = tf.unstack(tf.shape(name))

        self.batch1={'batch_size':batch_size_}
        self.seq_len_1={'sequence_length':seq_len_}

        # actual length of sequences considering padding
        seqlens = tf.count_nonzero(name, axis=-1)

        self.without_zero={'without_zero':seqlens}

        # word embedding
        wemb = tf.get_variable(shape=[vocab_size, wdim], dtype=tf.float32,
                               initializer=tf.random_uniform_initializer(-0.01, 0.01),
                               name='word_embedding')

        self.wemb3={'wemb':wemb}
        self.lookup={'lookup':tf.nn.embedding_lookup(wemb,name)}

        with tf.variable_scope('encoder') as scope:
            super12 = tf.nn.bidirectional_dynamic_rnn(
                tf.nn.rnn_cell.LSTMCell(hdim),
                tf.nn.rnn_cell.LSTMCell(hdim),
                inputs=tf.nn.embedding_lookup(wemb, name),
                sequence_length=seqlens,
                dtype=tf.float32)

        self.lstm_output={'output':super12}

        _, (fsf, fsb) =super12

        # output projection parameters
        Wo = tf.get_variable('Wo',
                             shape=[hdim * 2, num_labels],
                             dtype=tf.float32,
                             initializer=tf.random_uniform_initializer(-0.01, 0.01))

        self.weiths_wo={'weights20':Wo}

        concating_data=tf.concat([fsf.c,fsb.c],axis=-1)

        self.concating_visu={'concating_visualize':concating_data}
        #
        # bo = tf.get_variable('bo',
        #                      shape=[num_labels, ],
        #                      dtype=tf.float32,
        #                      initializer=tf.random_uniform_initializer(-0.01, 0.01))
        #
        # logits = tf.matmul(tf.concat([fsf.c, fsb.c], axis=-1), Wo) + bo
        #
        # probs = tf.nn.softmax(logits)
        # preds = tf.argmax(probs, axis=-1)
        #
        # # Cross Entropy
        # ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        # loss = tf.reduce_mean(ce)
        #
        # # Accuracy
        # accuracy = tf.reduce_mean(
        #     tf.cast(
        #         tf.equal(tf.cast(preds, tf.int32), labels),
        #         tf.float32
        #     )
        # )
        #
        # self.out = {
        #     'loss': loss,
        #     'prob': probs,
        #     'pred': preds,
        #     'logits': logits,
        #     'accuracy': accuracy
        # }
        #
        # # training operation
        # self.trainop = tf.train.AdamOptimizer().minimize(loss)


def rand_exec(model):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print('weightswo',model.weiths_wo)
        return sess.run([model.without_zero,model.wemb3,model.lstm_output,model.batch1,model.seq_len_1,model.lookup,model.concating_visu,model.weiths_wo],
                        feed_dict={
                            model.placeholders['name']: _placeholder_value,
                            model.placeholders['label']: np.random.randint(0, 10, [8, ])
                        }
                        )


if __name__ == '__main__':
    model = BasicClassifier(4,4,4,4)
    out = rand_exec(model)

    x,y,z,batch11,seq11,lookup32,con,weights1=out
    print(x,'\n\n')
    print(y,'\n\n')
    zz,(xx,yy)=z['output']


    print('\n\n\n\n\n')
    print("zz is here",zz,'\n\n\n\n')
    print(zz[0].shape)



    print("xx is here",xx,'\n\n\n\n')

    print("yy is here",yy,'\n\n\n\n')

    print("batch11",batch11,'\n\n\n\n')
    print('sequence is',seq11,'\n\n\n\n')

    print('lookup is',lookup32,'\n\n\n\n')

    print(lookup32['lookup'].shape)

    print('con',con)
    print("weights",weights1)
    
    
    
    
    
    
    
    
    
    #output:
    
    /anaconda/bin/python /Users/exepaul/Desktop/pratice_with_tensor_aaditya/exercises/ex_1_classify_names/model_basic_classifier.py
/anaconda/lib/python3.5/importlib/_bootstrap.py:222: RuntimeWarning: compiletime version 3.6 of module 'tensorflow.python.framework.fast_tensor_util' does not match runtime version 3.5
  return f(*args, **kwds)
[[1 1 1 0]
 [0 0 1 1]]
(2, 4)
2018-03-24 10:38:03.889087: I tensorflow/core/platform/cpu_feature_guard.cc:140] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
weightswo {'weights20': <tf.Variable 'Wo:0' shape=(8, 4) dtype=float32_ref>}
{'without_zero': array([3, 2])} 


{'wemb': array([[ 0.00233371,  0.00047011, -0.00290235, -0.00171303],
       [ 0.0080775 , -0.00371759,  0.00980962,  0.0083936 ],
       [-0.00082523, -0.00783622,  0.00786122, -0.00925037],
       [ 0.00017705, -0.00924235,  0.00275822, -0.00659752]],
      dtype=float32)} 








zz is here (array([[[-0.00115179, -0.00041491,  0.00109532, -0.00064341],
        [-0.00195484, -0.0008336 ,  0.00172486, -0.00104617],
        [-0.00251372, -0.00121351,  0.00208844, -0.0013007 ],
        [ 0.        ,  0.        ,  0.        ,  0.        ]],

       [[ 0.00016996, -0.00016206, -0.00022991,  0.00061312],
        [ 0.00029055, -0.00018982, -0.00033271,  0.00104509],
        [ 0.        ,  0.        ,  0.        ,  0.        ],
        [ 0.        ,  0.        ,  0.        ,  0.        ]]],
      dtype=float32), array([[[ 7.1868696e-04, -1.2120042e-03,  9.0654701e-04, -1.9700294e-03],
        [ 4.8621802e-04, -1.0339820e-03,  6.5511669e-04, -1.5082272e-03],
        [ 2.3480931e-04, -6.6985196e-04,  3.5326442e-04, -8.6288468e-04],
        [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00]],

       [[ 7.9076219e-04,  7.8381074e-04, -2.0581420e-04,  6.0351656e-05],
        [ 4.8068209e-04,  4.5061790e-04, -1.0755018e-04,  3.5494352e-06],
        [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00],
        [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00]]],
      dtype=float32)) 




(2, 4, 4)
xx is here LSTMStateTuple(c=array([[-0.00501123, -0.0024302 ,  0.00417222, -0.00259759],
       [ 0.00058115, -0.00037968, -0.00066556,  0.00209053]],
      dtype=float32), h=array([[-0.00251372, -0.00121351,  0.00208844, -0.0013007 ],
       [ 0.00029055, -0.00018982, -0.00033271,  0.00104509]],
      dtype=float32)) 




yy is here LSTMStateTuple(c=array([[ 0.00144089, -0.00242214,  0.00181399, -0.00393816],
       [ 0.00158017,  0.00156869, -0.00041194,  0.00012077]],
      dtype=float32), h=array([[ 7.1868696e-04, -1.2120042e-03,  9.0654701e-04, -1.9700294e-03],
       [ 7.9076219e-04,  7.8381074e-04, -2.0581420e-04,  6.0351656e-05]],
      dtype=float32)) 




batch11 {'batch_size': 2} 




sequence is {'sequence_length': 4} 




lookup is {'lookup': array([[[ 0.0080775 , -0.00371759,  0.00980962,  0.0083936 ],
        [ 0.0080775 , -0.00371759,  0.00980962,  0.0083936 ],
        [ 0.0080775 , -0.00371759,  0.00980962,  0.0083936 ],
        [ 0.00233371,  0.00047011, -0.00290235, -0.00171303]],

       [[ 0.00233371,  0.00047011, -0.00290235, -0.00171303],
        [ 0.00233371,  0.00047011, -0.00290235, -0.00171303],
        [ 0.0080775 , -0.00371759,  0.00980962,  0.0083936 ],
        [ 0.0080775 , -0.00371759,  0.00980962,  0.0083936 ]]],
      dtype=float32)} 




(2, 4, 4)
con {'concating_visualize': array([[-0.00501123, -0.0024302 ,  0.00417222, -0.00259759,  0.00144089,
        -0.00242214,  0.00181399, -0.00393816],
       [ 0.00058115, -0.00037968, -0.00066556,  0.00209053,  0.00158017,
         0.00156869, -0.00041194,  0.00012077]], dtype=float32)}
weights {'weights20': array([[ 0.00088152, -0.0001231 , -0.00120286,  0.00778185],
       [-0.00494761, -0.00051359, -0.00741149, -0.00169441],
       [-0.00236187, -0.0095766 ,  0.00066017,  0.0057791 ],
       [-0.00404736, -0.00276053,  0.00982949, -0.00990209],
       [-0.00543521, -0.00328956,  0.00813769,  0.00674003],
       [ 0.00783077,  0.00736379, -0.0075415 ,  0.00172279],
       [-0.00112041, -0.00112254, -0.00957669, -0.00545661],
       [ 0.00192218, -0.00214659,  0.00154967,  0.00519598]],
