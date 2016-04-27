
import time

import tensorflow as tf

import input_data
import graph

import matplotlib
matplotlib.use('Agg')  # http://stackoverflow.com/a/4706614
import matplotlib.pyplot as plt


VERSION = '1.2'

LEARNING_RATE = 1e-4

DROPOUT = 1.0


def placeholder_inputs(batch_size):
    images_placeholder = tf.placeholder(
        tf.float32,
        shape = (batch_size, input_data.WINDOW_SIZE, input_data.WINDOW_SIZE, 3)
    )
    labels_placeholder = tf.placeholder(
        tf.int64,
        shape = (batch_size)
    )
    keep_prob_placeholder = tf.placeholder(
        tf.float32
    )
    return images_placeholder, labels_placeholder, keep_prob_placeholder


def run_training():

    training_dataset, test_dataset = input_data.get_datasets()
    training_dataqueue = input_data.make_dataqueue(training_dataset)

    with tf.Graph().as_default():

        optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)

        images_pl, labels_pl, keep_prob_pl = placeholder_inputs(training_dataset.batch_size)
        logits      = graph.inference(images_pl, keep_prob_pl)
        loss        = graph.loss(logits, labels_pl)
        train_op    = graph.training(optimizer, loss)
        evaluation  = graph.evaluation(logits, labels_pl)

        saver = tf.train.Saver()

        sess = tf.Session()
        init = tf.initialize_all_variables()
        sess.run(init)

        def save_model(step, eval_value):
            saver.save(
                sess,
                '/home/hannah/data/mcam-artifacts/saved-model-v%s-s%d-e%f'  \
                    % (VERSION, step, eval_value)
            )

        step = 0
        trace = []
        t0 = time.time()
        last_checkpoint = t0
        last_print      = t0
        last_plot       = t0
        while True:
            step += 1
            start_time = time.time()

            if training_dataqueue.empty():
                print 'Training queue empty! Waiting...'
            batch = training_dataqueue.get()
            if batch is None:
                print 'Training queue exhausted'
                save_model(step, eval_value)
                break

            images_feed, labels_feed = batch
            feed_dict = {
                images_pl: images_feed,
                labels_pl: labels_feed,
                keep_prob_pl: DROPOUT
            }
            _, loss_value, eval_value = sess.run(
                [train_op, loss, evaluation],
                feed_dict=feed_dict
            )

            end_time = time.time()
            trace.append((step, loss_value, eval_value))

            if end_time - last_print > 2:
                print ('Step %d:  cross-entropy = %.2f  accuracy = %.2f  time = %.2f'  \
                           % (step, loss_value, eval_value, end_time-start_time))
                last_print = time.time()

            #### DEBUG
            if end_time - last_plot > 10:
                plt.figure('trace_loss')
                plt.clf()
                plt.plot([s for s,l,e in trace], [l for s,l,e in trace])
                plt.title('Cross-entropy')
                plt.savefig('debug/loss.png')

                plt.figure('trace_eval')
                plt.clf()
                plt.plot([s for s,l,e in trace], [e for s,l,e in trace])
                plt.title('Accuracy')
                plt.savefig('debug/eval.png')

                last_plot = time.time()
            ####

            if end_time - last_checkpoint > 5*60:
                print '\n ==== Saving checkpoint ==== \n'
                save_model(step, eval_value)
                last_checkpoint = time.time()


def main(_):
    run_training()


if __name__ == '__main__':
  tf.app.run()




