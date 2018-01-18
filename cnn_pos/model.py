import numpy as np
import tensorflow as tf
import pickle


class config():
  sequence_length = 31
  batch_size = 50
  vocab_size = 16314
  triger_size = 34
  position_embedded_size = 50
  embedding_size = 300
  filter_sizes = [2,3,4]
  feature_size = 150
  

class POS_CNN_Model(object):
  def __init__(
      self, config, l2_reg_lambda=0.003):
    self.config = config
    self.input_x = tf.placeholder(tf.int32,
                                  [None, self.config.sequence_length], name="input_x")
    self.input_y = tf.placeholder(tf.float32,
                                  [None, self.config.triger_size],
                                  name="input_y")
    self.size_batch = tf.placeholder(tf.int32, name="size_batch")
    self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
    
    self.feature = self.add_embedding(self.config.vocab_size)
    self.add_model(l2_reg_lambda)


  def add_embedding(self, vocab_size):
    with tf.variable_scope('embedded_layer'):
      WV = tf.get_variable('word_vectors', 
                           initializer=tf.random_uniform([self.config.vocab_size, self.config.embedding_size], -1.0, 1.0), 
                           dtype=tf.float32)
      wv = tf.nn.embedding_lookup(WV, self.input_x)
      postion_embedding = tf.get_variable("postion_embedding",
                                          shape=[self.config.sequence_length,
                                                 self.config.position_embedded_size],
                                          dtype=tf.float32)
      wv = [tf.squeeze(x, [1]) for x in tf.split(wv, self.config.sequence_length, 1)]
      inputs = []
      for v in range(len(wv)):
        inputs.append(tf.concat((wv[v],
                                 tf.reshape(tf.tile(postion_embedding[v], [self.size_batch]),
                                          [self.size_batch, self.config.position_embedded_size])),
                                1))
      inputs = tf.transpose(tf.stack(inputs), perm=[1,0,2])
      return tf.expand_dims(inputs, [-1])
          

  def add_model(self, l2_reg_lambda):
    """
    :param l2_reg_lambda:
            self.input_x: list of tensor len = sentence_length, each tensor has
                shape = [batch_size, embed_size]
    :return:
    """
    # Create a convolution + maxpool layer for each filter size
    num_filters_total = 0
    pooled_outputs = []
    W = []
    b = []
    postion_embedding = []
    for  filter_size in self.config.filter_sizes:
      with tf.name_scope("Conv-maxpool-%s" % filter_size):
        # Convolution Parameter
        filter_shape = [filter_size,
                        self.config.embedding_size + self.config.position_embedded_size,
                        1, self.config.feature_size]
        W.append(tf.get_variable("W_%d" %filter_size, 
                                 shape=filter_shape,
                                 dtype=tf.float32))
        b.append(tf.get_variable("b_%d" %filter_size, 
                                 shape=[self.config.feature_size],
                                 dtype=tf.float32))
 
        # Convolution Layer                                
        conv = tf.nn.conv2d(
          self.feature,
          W[-1],
          strides=[1, 1, 1, 1],
          padding="VALID",
          name="conv")
        
        # Apply nonlinearity
        h = tf.nn.relu(tf.nn.bias_add(conv, b[-1]), name="relu")
        tf.add_to_collection("loss", tf.nn.l2_loss(W[-1]) + tf.nn.l2_loss(b[-1]))
        
        # Max-pooling over the outputs
        pooled = tf.nn.max_pool(
          h,
          ksize=[1, self.config.sequence_length - filter_size + 1, 1, 1],
          strides=[1, 1, 1, 1],
          padding='VALID',
          name="pool")
        pooled_outputs.append(pooled)
        

        # Combine all the pooled features
    num_filters_total = self.config.feature_size * len(self.config.filter_sizes)
    triger_size = self.config.triger_size
    self.h_pool = tf.concat(pooled_outputs, 2)
    self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
    with tf.name_scope("dropout"):
      self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
    with tf.name_scope("output"):
      W2 = tf.get_variable(
        "W2",
        shape=[ num_filters_total, self.config.triger_size])
      b2 = tf.get_variable(name="b2", shape=[self.config.triger_size], 
                           dtype=tf.float32)
      tf.add_to_collection("loss", tf.nn.l2_loss(W2) + tf.nn.l2_loss(b2))
      self.scores = tf.nn.xw_plus_b(self.h_drop, W2, b2, name="scores")
      self.predictions = tf.argmax(self.scores, 1, name="predictions")
      
      # CalculateMean cross-entropy loss
    with tf.name_scope("loss"):
      losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
      self.loss = tf.reduce_mean(losses) + l2_reg_lambda * tf.add_n(tf.get_collection("loss")) / 2
      
      # Accuracy
    with tf.name_scope("accuracy"):
      correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
      self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
          
