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
