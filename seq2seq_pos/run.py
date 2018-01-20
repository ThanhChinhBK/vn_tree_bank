import tensorflow as tf
from model import *
import numpy as np
import os
import dill
from data_utils import get_full_vocab, data2encode
import time
import datetime


tf.flags.DEFINE_integer("batch_size", 128, "batch size")
tf.flags.DEFINE_integer("n_epochs", 100, "num epochs")
tf.flags.DEFINE_float("learning_rate", 0.001, "learning rate")
tf.flags.DEFINE_integer("embedding_dim", 300, "word embedding dimession")
tf.flags.DEFINE_integer("hidden_dim", 300, "hidden vector of lstm")
tf.flags.DEFINE_string("data_path", "../BKTreebank_LREC2018", "data path")
tf.flags.DEFINE_integer("label_embedding_size", 100, "label embdding size")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_string("out_dir", "runs/", "path to save checkpoint")
tf.flags.DEFINE_integer("window_size", 7 , "window size")
tf.flags.DEFINE_integer("dropout", 0.1, "dropout")

tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
FLAGS = tf.flags.FLAGS

def batch(inputs, outputs, max_sequence_length=None):
  """
  Args:
      inputs:
          list of sentences (integer lists)
      max_sequence_length:
          integer specifying how large should `max_time` dimension be.
          If None, maximum sequence length would be used
    
  Outputs:
      inputs_time_major:
          input sentences transformed into time-major matrix 
          (shape [max_time, batch_size]) padded with 0s
      sequence_lengths:
          batch-sized list of integers specifying amount of active 
          time steps in each input sequence
  """
    
  sequence_lengths = [len(seq) for seq in inputs]
  batch_size = len(inputs)
    
  if max_sequence_length is None:
    max_sequence_length = max(sequence_lengths)
    
  inputs_batch_major = np.zeros(shape=[batch_size, max_sequence_length], dtype=np.int32) # == PAD 
  outputs_batch_major = np.zeros(shape=[batch_size, max_sequence_length], dtype=np.int32) # == PAD 
    
  for i, (in_seq, out_seq) in enumerate(zip(inputs, outputs)):
    for j, (in_element, out_element) in enumerate(zip(in_seq, out_seq)):
      inputs_batch_major[i, j] = in_element
      outputs_batch_major[i, j] = out_element

    # [batch_size, max_time] -> [max_time, batch_size]
  inputs_time_major = inputs_batch_major.swapaxes(0, 1)
  outputs_time_major = outputs_batch_major.swapaxes(0,1)

  return inputs_time_major, outputs_time_major, sequence_lengths


def batch_iter(full_data, full_labels, batch_size):
  """
  Args:
      full_data:
          list of sentences (integer lists)
      batch_size:
          size of one batch
    
  Outputs:
      sentences:
          list of sentences in one batch
      sentences_length:
          length of each sentence in batch
  """
  total_batch = len(full_data) // batch_size
  for i in range(total_batch + 1):
    batch_sentences = full_data[i:i+batch_size]
    batch_labels = full_labels[i:i+batch_size]
    sentences, labels, sentences_length = batch(batch_sentences, full_labels)
    yield sentences, labels, sentences_length

if __name__ == "__main__":
  word_vocab, label_vocab = get_full_vocab(
    [os.path.join(FLAGS.data_path,x) for x in ["test03_cleaned", "train03-train_cleaned", "train03-dev_cleaned"]])

  train_word_windows, train_addition_windows, train_label_windows = data2encode(
    os.path.join(FLAGS.data_path, "train03-train_cleaned"), word_vocab, label_vocab, window_size=FLAGS.window_size)
  dev_word_windows, dev_addition_windows, dev_label_windows = data2encode(
    os.path.join(FLAGS.data_path, "train03-dev_cleaned"), word_vocab, label_vocab, window_size=FLAGS.window_size)
  test_word_windows, test_addition_windows, test_label_windows = data2encode(
    os.path.join(FLAGS.data_path, "test03_cleaned"), word_vocab, label_vocab, window_size=FLAGS.window_size)

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
  
  
  with tf.Session(config=session_conf) as sess:
    pos_seq2seq = LSTMAutoEncoder(len(word_vocab),
                                  len(label_vocab),
                                  FLAGS.hidden_dim,
                                  FLAGS.embedding_dim,
                                  FLAGS.label_embedding_size,
                                  FLAGS.learning_rate)
    print("model created")

    global_step = tf.Variable(0, name="global_step", trainable=False)
    increment_global_step_op = tf.assign(global_step, global_step+1)
    train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(pos_seq2seq.loss)

    loss_summary = tf.summary.scalar("loss", pos_seq2seq.loss)
    acc_summary = tf.summary.scalar("accuracy", pos_seq2seq.accuracy)  

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
    
    def train_step(model, sess, x_batch, y_batch, x_length, epoch):
      feed_dict = {
        model.encoder_inputs : x_batch,
        model.decoder_inputs : y_batch,
        model.encoder_inputs_length : x_length
      }
      _, loss, step, summary = sess.run([model.train_op, model.loss,  increment_global_step_op, train_summary_op]
                                        , feed_dict=feed_dict)
      time_str = datetime.datetime.now().isoformat()
      print("{}: epoch:{} step {}, loss {:g}".format(time_str, epoch, step, loss))
      train_summary_writer.add_summary(summary, step)
      
    def dev_step(model, sess, x_dev, y_dev,epoch):
      x_dev_time, y_dev_time, x_dev_length = batch(x_dev, y_dev)
      feed_dict = {
        model.encoder_inputs : x_dev_time,
        model.decoder_inputs : y_dev_time,
        model.encoder_inputs_length : x_dev_length 
      }
      
      loss, acc,step, summary = sess.run([model.loss, model.accuracy,  increment_global_step_op, dev_summary_op]
                                        , feed_dict=feed_dict)
      time_str = datetime.datetime.now().isoformat()
      print("DEVELOPE:")
      print("{}: epoch:{} step {}, loss {:g}, acc {:g}".format(time_str, epoch, step, loss, acc))
      dev_summary_writer.add_summary(summary, step)
      
    for e in np.arange(FLAGS.n_epochs):
      for x_batch, y_batch, x_length in batch_iter(train_word_windows, train_label_windows, FLAGS.batch_size):
        train_step(pos_seq2seq, sess, x_batch, y_batch,x_length,  e)
      dev_step(pos_seq2seq, sess, dev_word_windows, dev_label_windows, e)
      save_path = saver.save(sess, os.path.join(checkpoint_dir, "checkpoint"), e)
      print("Model saved in file: %s" % save_path)
