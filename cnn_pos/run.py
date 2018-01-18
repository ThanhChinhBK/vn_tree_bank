import tensorflow as tf
from model import *
import numpy as np
import os
import dill
from data_utils import get_full_vocab, data2window
import time
import datetime


tf.flags.DEFINE_integer("batch_size", 128, "batch size")
tf.flags.DEFINE_integer("n_epochs", 100, "num epochs")
tf.flags.DEFINE_float("learning_rate", 0.001, "learning rate")
tf.flags.DEFINE_integer("embedding_dim", 300, "word embedding dimession")
tf.flags.DEFINE_integer("num_features", 100, "features vector after cnn")
tf.flags.DEFINE_string("data_path", "../BKTreebank_LREC2018", "data path")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_string("out_dir", "runs/", "path to save checkpoint")
tf.flags.DEFINE_integer("window_size", 7 , "window size")
tf.flags.DEFINE_integer("dropout", 0.1, "dropout")

tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
FLAGS = tf.flags.FLAGS

def onehotencode(arr, num_values):
  b = np.zeros((len(arr), num_values))
  b[np.arange(len(arr)), arr] = 1
  return b

def generate_batch(word_windows, label_windows):
  total_index = len(word_windows) // FLAGS.batch_size
  for i in np.arange(total_index + 1):
    yield word_windows[FLAGS.batch_size * i: FLAGS.batch_size * (i+1)], label_windows[FLAGS.batch_size * i: FLAGS.batch_size * (i+1)]
  yield word_windows[FLAGS.batch_size * total_index:], label_windows[FLAGS.batch_size * total_index:]

if __name__ == "__main__":
  word_vocab, label_vocab = get_full_vocab(
    [os.path.join(FLAGS.data_path,x) for x in ["test03_cleaned", "train03-train_cleaned", "train03-dev_cleaned"]])

  train_word_windows, train_addition_windows, train_label_windows = data2window(
    os.path.join(FLAGS.data_path, "train03-train_cleaned"), word_vocab, label_vocab, window_size=FLAGS.window_size)
  train_label_onehot = onehotencode(train_label_windows, len(label_vocab))
  dev_word_windows, dev_addition_windows, dev_label_windows = data2window(
    os.path.join(FLAGS.data_path, "train03-dev_cleaned"), word_vocab, label_vocab, window_size=FLAGS.window_size)
  dev_label_onehot = onehotencode(dev_label_windows, len(label_vocab))
  test_word_windows, test_addition_windows, test_label_windows = data2window(
    os.path.join(FLAGS.data_path, "test03_cleaned"), word_vocab, label_vocab, window_size=FLAGS.window_size)
  test_label_onehot = onehotencode(test_label_windows, len(label_vocab))

  print("length of vocab:", len(word_vocab))
  print("length of train data:", len(train_word_windows))
  print("length of dev data:", len((dev_word_windows)))
  print("length of test data:", len((test_word_windows)))

  if not os.path.exists(FLAGS.out_dir):
    os.makedirs(FLAGS.out_dir)
  checkpoint_dir = os.path.abspath(os.path.join(FLAGS.out_dir, "checkpoints"))
  dill.dump(word_vocab,open(os.path.join(FLAGS.out_dir, "dictionary.pkl"),'wb')) 
  tf.reset_default_graph()
  session_conf = tf.ConfigProto(
    allow_soft_placement=FLAGS.allow_soft_placement,
    log_device_placement=FLAGS.log_device_placement)
  
  conf = config()
  config.sequence_length = FLAGS.window_size * 2 + 1
  config.feature_size = FLAGS.num_features
  config.triger_size = len(label_vocab)
  config.embedding_size = FLAGS.embedding_dim

  with tf.Session(config=session_conf) as sess:
    pos_cnn = POS_CNN_Model(conf)
    print("model created")

    global_step = tf.Variable(0, name="global_step", trainable=False)
    increment_global_step_op = tf.assign(global_step, global_step+1)
    train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(pos_cnn.loss)

    loss_summary = tf.summary.scalar("loss", pos_cnn.loss)
    acc_summary = tf.summary.scalar("accuracy", pos_cnn.accuracy)  

    # Train Summaries
    train_summary_op = tf.summary.merge([loss_summary])
    train_summary_dir = os.path.join(FLAGS.out_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

    # Dev summaries
    dev_summary_op = tf.summary.merge([loss_summary])
    dev_summary_dir = os.path.join(FLAGS.out_dir, "summaries", "dev")
    dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)
    
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)
    

    sess.run(tf.global_variables_initializer())
    
    def train_step(model, sess, x_batch, y_batch, epoch):
      feed_dict = {
        model.input_x : x_batch,
        model.input_y : y_batch,
        model.size_batch : len(x_batch),
        model.dropout_keep_prob : FLAGS.dropout
      }
      _, loss, step, summary = sess.run([train_op, model.loss,  increment_global_step_op, train_summary_op]
                                        , feed_dict=feed_dict)
      time_str = datetime.datetime.now().isoformat()
      print("{}: epoch:{} step {}, loss {:g}".format(time_str, epoch, step, loss))
      train_summary_writer.add_summary(summary, step)
      
    def dev_step(model, sess, x_dev, y_dev, epoch):
      feed_dict = {
        model.input_x : x_dev,
        model.input_y : y_dev,
        model.size_batch : len(x_dev),
        model.dropout_keep_prob : FLAGS.dropout
      }
      
      loss, acc,step, summary = sess.run([model.loss, model.accuracy,  increment_global_step_op, dev_summary_op]
                                        , feed_dict=feed_dict)
      time_str = datetime.datetime.now().isoformat()
      print("DEVELOPE:")
      print("{}: epoch:{} step {}, loss {:g}, acc {:g}".format(time_str, epoch, step, loss, acc))
      dev_summary_writer.add_summary(summary, step)
      
    for e in np.arange(FLAGS.n_epochs):
      for x_batch, y_batch in generate_batch(train_word_windows, train_label_onehot):
        train_step(pos_cnn, sess, x_batch, y_batch, e)
      dev_step(pos_cnn, sess, dev_word_windows, dev_label_onehot, e)
      save_path = saver.save(sess, os.path.join(checkpoint_dir, "checkpoint"), e)
      print("Model saved in file: %s" % save_path)
