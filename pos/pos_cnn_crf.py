import tensorflow as tf
import utils as utils
#import aux_network_func as af
import data_utils as du
import pos_model as pm

import dill

import numpy as np
import os
import time
import datetime
from tensorflow.python import debug as tf_debug

#tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)") #not used 
tf.flags.DEFINE_string("train_path", "../BKTreebank_LREC2018/train03-train", "Train Path")
tf.flags.DEFINE_string("test_path", "../BKTreebank_LREC2018/test03", "Test Path")
tf.flags.DEFINE_string("dev_path", "../BKTreebank_LREC2018/train03-dev", "dev Path")
tf.flags.DEFINE_string("out_dir", "../precessing", "out dir")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("grad_clip", 5, 
                      "value for gradient clipping to avoid exploding/vanishing gradient(default: 5.0) in LSTM")
tf.flags.DEFINE_float("max_global_clip", 5.0, 
                      "value for gradient clipping to avoid exploding/vanishing gradient overall(default: 1.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 10, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("word_col", 1, "position of the word in input file (default: 0)")
tf.flags.DEFINE_integer("label_col", 3, "position of the label in input file (default: 3)")
tf.flags.DEFINE_integer("n_hidden_LSTM", 200, "Number of hidden units in LSTM (default: 200)")
tf.flags.DEFINE_integer("num_epochs", 50, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("num_filters", 30, "Number of filters to apply for char CNN (default: 30)") 
tf.flags.DEFINE_integer("filter_size", 3, "filter_size (default: 3 )")
tf.flags.DEFINE_integer("evaluate_every", 1000, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("word_embedd_dim", 300, "word embedd dum(default:300")
tf.flags.DEFINE_integer("char_embedd_dim", 50, "char_embedd_dim(default: 50)")
tf.flags.DEFINE_integer("Optimizer", 1, "Adam : 1 , SGD:2")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_float("starter_learning_rate", 0.015, "Initial learning rate for the optimizer. (default: 1e-3)")
tf.flags.DEFINE_float("decay_rate", 0.05, "How much to decay the learning rate. (default: 0.015)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_boolean("PadZeroBegin", False, "where to pad zero in the input")
FLAGS = tf.flags.FLAGS


word_alphabet = ['<PAD>']
label_alphabet = ['<PAD>']
print("Reading data from (train, test, dev) set...")
word_sentences_train, _, word_index_sentences_train, label_index_sentences_train = du.read_conll_sequence_labeling(
    FLAGS.train_path, word_alphabet, label_alphabet, FLAGS.word_col, FLAGS.label_col)
word_sentences_dev, _, word_index_sentences_dev, label_index_sentences_dev = du.read_conll_sequence_labeling(
    FLAGS.dev_path, word_alphabet, label_alphabet, FLAGS.word_col, FLAGS.label_col)
word_sentences_test, _, word_index_sentences_test, label_index_sentences_test = du.read_conll_sequence_labeling(
    FLAGS.test_path, word_alphabet, label_alphabet, FLAGS.word_col, FLAGS.label_col, FLAGS.out_dir)
print("dictionary size is %d, label size is %d" %(len(word_alphabet), len(label_alphabet)))
max_length_train = utils.get_max_length(word_sentences_train)
max_length_dev = utils.get_max_length(word_sentences_dev)
#max_length_test = utils.get_max_length(word_sentences_test)
max_length = min(du.MAX_LENGTH, max(max_length_train, max_length_dev))

print("Maximum length (i.e max words ) of training set is %d" % max_length_train)
print("Maximum length (i.e max words ) of dev set is %d" % max_length_dev)
print("Maximum length (i.e max words ) used for training is %d" % max_length)

print("Padding training text and lables ...")
word_index_sentences_train_pad,train_seq_length = du.padSequence(word_index_sentences_train,
                                                                 max_length)
label_index_sentences_train_pad,_= du.padSequence(label_index_sentences_train,max_length)

print("Padding dev text and lables ...")
word_index_sentences_dev_pad,dev_seq_length = du.padSequence(word_index_sentences_dev,max_length)
label_index_sentences_dev_pad,_= du.padSequence(label_index_sentences_dev,max_length)

print("Creating character set FROM training set ...")
char_alphabet = ['<PAD>']
char_index_train,max_char_per_word_train= du.generate_character_data(word_sentences_train,  
                                                                     char_alphabet=char_alphabet,
                                                                     setType="Train")
print("Creating character set FROM dev set ...")
char_index_dev,max_char_per_word_dev= du.generate_character_data(word_sentences_dev, 
                                    char_alphabet=char_alphabet, setType="Dev")


print("character alphabet size: %d" % (len(char_alphabet) - 1))
max_char_per_word = min(du.MAX_CHAR_PER_WORD, max_char_per_word_train,max_char_per_word_dev)
print("Maximum character length is %d" %max_char_per_word)
print("Padding Training set ...")
char_index_train_pad = du.construct_padded_char(char_index_train, char_alphabet, 
                                                max_sent_length=max_length,max_char_per_word=max_char_per_word)
print("Padding Dev set ...")
char_index_dev_pad = du.construct_padded_char(char_index_dev, char_alphabet, 
                                              max_sent_length=max_length,max_char_per_word=max_char_per_word)
checkpoint_dir = os.path.abspath(os.path.join(FLAGS.out_dir, "checkpoints"))
dill.dump(char_alphabet,open(os.path.join(FLAGS.out_dir, "char_alphabet.pkl"),'wb')) 
dill.dump(label_alphabet,open(os.path.join(FLAGS.out_dir, "label_alphabet.pkl"),'wb')) 
tf.reset_default_graph()

session_conf = tf.ConfigProto(
  allow_soft_placement=FLAGS.allow_soft_placement,
  log_device_placement=FLAGS.log_device_placement)

with tf.Session(config=session_conf) as sess:
  best_accuracy = 0 
  best_overall_accuracy = 0
  best_accuracy_test = 0 
  best_overall_accuracy_test = 0
  best_step = 0
  BiLSTM = pm.textBiLSTM(sequence_length=max_length, num_classes=len(label_alphabet), 
                         word_vocab_size=len(word_alphabet),word_embedd_dim=FLAGS.word_embedd_dim,
                         n_hidden_LSTM =FLAGS.n_hidden_LSTM,max_char_per_word=max_char_per_word,
                         char_vocab_size=len(char_alphabet),char_embedd_dim = FLAGS.char_embedd_dim,
                         grad_clip=FLAGS.grad_clip,num_filters=FLAGS.num_filters,
                         filter_size= FLAGS.filter_size)
  print("model ceated")
