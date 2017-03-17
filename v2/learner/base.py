
import tensorflow as tf


class Learner(object):

    '''
    Subclass must define:
        optimizer
        inputs_pl
        labels_pl
        outputs
        loss
        stats
    '''

    def __init__(self, savefile=None):
        self.global_step = tf.Variable(0, trainable=False)
        self.train_op = self.optimizer.minimize(
            self.loss, global_step=self.global_step
        )

        self.sess = tf.Session()
        self.saver = tf.train.Saver()

        if savefile:
            # self.saver.restore(self.sess, savefile)
            new_saver = tf.train.import_meta_graph(savefile)
            new_saver.restore(self.sess, savefile.split('.')[0])
        else:
            self.sess.run(tf.global_variables_initializer())

    def train(self, inputs, labels, feed_dict={}):
        _, stats = self.sess.run(
            [self.train_op, self.stats],
            feed_dict=dict({self.inputs_pl: inputs, self.labels_pl: labels}, **feed_dict)
        )
        return stats

    def test(self, inputs, labels, feed_dict={}):
        stats = self.sess.run(
            [self.stats],
            feed_dict=dict({self.inputs_pl: inputs, self.labels_pl: labels}, **feed_dict)
        )
        return stats

    def get_global_step(self):
        return self.sess.run(self.global_step)

    def __call__(self, inputs, feed_dict={}):
        outputs, m = self.sess.run(
            [self.outputs, self.merged],
            feed_dict=dict({self.inputs_pl: inputs}, **feed_dict)
        )
        self.writer.add_summary(m)
        return outputs









