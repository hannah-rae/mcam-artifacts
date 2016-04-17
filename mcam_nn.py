# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# pylint: disable=missing-docstring
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import time
from six.moves import xrange  # pylint: disable=redefined-builtin

# from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import input_data
import graph

# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 1e-4, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 91, 'Number of steps to run trainer.')
flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('batch_size', 100, 'Batch size.  '
                     'Must divide evenly into the dataset sizes.')



def get_next_batch(batch_size, batch_num, dataset):
    img_list = []
    label_list = []
    offset = batch_size * batch_num
    for i in range(batch_size):
        img = input_data.get_array(dataset[offset + i][0]).flatten()
        label = dataset[offset + i][1][1]
        img_list.append(img)
        label_list.append(label)
    return img_list, label_list

# def print_stats(feed_dict):
#     predicted_class = tf.argmax(y_predict, 1) # vector of (class# for each example)
#     actual_class = tf.argmax(y_actual, 1)

#     num_predicted_pos = tf.reduce_sum(predicted_class)
#     num_actual_pos = tf.reduce_sum(actual_class)
#     num_true_pos = tf.reduce_sum(tf.mul(predicted_class, actual_class))

#     num_predicted_pos_ = num_predicted_pos.eval(feed_dict=feed_dict)
#     num_actual_pos_ = num_actual_pos.eval(feed_dict=feed_dict)
#     num_true_pos_ = num_true_pos.eval(feed_dict=feed_dict)
#     try:
#         precision = float(num_true_pos_)/float(num_predicted_pos_)
#     except ZeroDivisionError:
#         precision = -1
#     try:
#         recall = float(num_true_pos_)/float(num_actual_pos_)
#     except ZeroDivisionError:
#         recall = -1
#     print "num_predicted: %d  num_actual: %d  num_both: %d  precision: %f  recall: %f" % \
#         (num_predicted_pos_, num_actual_pos_, num_true_pos_, precision, recall)

def placeholder_inputs(batch_size):
  """Generate placeholder variables to represent the input tensors.
  These placeholders are used as inputs by the rest of the model building
  code and will be fed from the downloaded data in the .run() loop, below.
  Args:
    batch_size: The batch size will be baked into both placeholders.
  Returns:
    images_placeholder: Images placeholder.
    labels_placeholder: Labels placeholder.
  """
  # Note that the shapes of the placeholders match the shapes of the full
  # image and label tensors, except the first dimension is now batch_size
  # rather than the full size of the train or test data sets.
  images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                         input_data.IMAGE_PIXELS))
  labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
  return images_placeholder, labels_placeholder


def fill_feed_dict(data_set, images_pl, labels_pl, step):
  """Fills the feed_dict for training the given step.
  A feed_dict takes the form of:
  feed_dict = {
      <placeholder>: <tensor of values to be passed for placeholder>,
      ....
  }
  Args:
    data_set: The set of images and labels, from input_data.read_data_sets()
    images_pl: The images placeholder, from placeholder_inputs().
    labels_pl: The labels placeholder, from placeholder_inputs().
  Returns:
    feed_dict: The feed dictionary mapping from placeholders to values.
  """
  # Create the feed_dict for the placeholders filled with the next
  # `batch size ` examples.
  images_feed, labels_feed = get_next_batch(FLAGS.batch_size, step, data_set)
  feed_dict = {
      images_pl: images_feed,
      labels_pl: labels_feed,
  }
  return feed_dict


def do_eval(sess,
            eval_correct,
            images_placeholder,
            labels_placeholder,
            data_set):
  """Runs one evaluation against the full epoch of data.
  Args:
    sess: The session in which the model has been trained.
    eval_correct: The Tensor that returns the number of correct predictions.
    images_placeholder: The images placeholder.
    labels_placeholder: The labels placeholder.
    data_set: The set of images and labels to evaluate, from
      input_data.read_data_sets().
  """
  # And run one epoch of eval.
  true_count = 0  # Counts the number of correct predictions.
  steps_per_epoch = len(data_set) / FLAGS.batch_size
  print steps_per_epoch
  num_examples = steps_per_epoch * FLAGS.batch_size
  for step in xrange(steps_per_epoch):
    feed_dict = fill_feed_dict(data_set,
                               images_placeholder,
                               labels_placeholder,
                               step)
    true_count += sess.run(eval_correct, feed_dict=feed_dict)
  precision = float(true_count) / float(num_examples)
  print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
        (num_examples, true_count, precision))


def run_training():
  # Gets the training and test data lists from input_data
  train_data, test_data = input_data.get_data()

  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default():
    # Generate placeholders for the images and labels.
    images_placeholder, labels_placeholder = placeholder_inputs(
        FLAGS.batch_size)

    # Build a Graph that computes predictions from the inference model.
    logits = graph.inference(images_placeholder,
                             FLAGS.hidden1,
                             FLAGS.hidden2)

    # Add to the Graph the Ops for loss calculation.
    loss = graph.loss(logits, labels_placeholder)

    # Add to the Graph the Ops that calculate and apply gradients.
    train_op = graph.training(loss, FLAGS.learning_rate)

    # Add the Op to compare the logits to the labels during evaluation.
    eval_correct = graph.evaluation(logits, labels_placeholder)

    # Build the summary operation based on the TF collection of Summaries.
    # summary_op = tf.merge_all_summaries()

    # Create a saver for writing training checkpoints.
    # saver = tf.train.Saver()

    # Create a session for running Ops on the Graph.
    sess = tf.Session()

    # Run the Op to initialize the variables.
    init = tf.initialize_all_variables()
    sess.run(init)

    # Instantiate a SummaryWriter to output summaries and the Graph.
    # summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

    # And then after everything is built, start the training loop.
    for step in xrange(FLAGS.max_steps):
      start_time = time.time()

      # Fill a feed dictionary with the actual set of images and labels
      # for this particular training step.
      feed_dict = fill_feed_dict(train_data,
                                 images_placeholder,
                                 labels_placeholder,
                                 step)

      # Run one step of the model.  The return values are the activations
      # from the `train_op` (which is discarded) and the `loss` Op.  To
      # inspect the values of your Ops or variables, you may include them
      # in the list passed to sess.run() and the value tensors will be
      # returned in the tuple from the call.
      _, loss_value = sess.run([train_op, loss],
                               feed_dict=feed_dict)

      duration = time.time() - start_time



      # Write the summaries and print an overview fairly often.
      if step % 1 == 0:
        print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))



      if (step + 1) % 100 == 0 or (step + 1) == FLAGS.max_steps:

        print('Training Data Eval:')
        do_eval(sess,
                eval_correct,
                images_placeholder,
                labels_placeholder,
                train_data)

        print('Test Data Eval:')
        do_eval(sess,
                eval_correct,
                images_placeholder,
                labels_placeholder,
                test_data)


def main(_):
  run_training()


if __name__ == '__main__':
  tf.app.run()