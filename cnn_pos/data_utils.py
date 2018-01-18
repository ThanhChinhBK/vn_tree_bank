import numpy as np
import nltk 
import os

def corpus_reader(file_path):
  sentences = []
  with open(file_path) as f:
    for line in f:
      sentences.append([nltk.tag.str2tuple(token) for token in line.strip().split(" ")])
  return sentences

def vocab_encode(sentences):
  vocab = {"<UNK>"}
  label = set()
  for sent in sentences:
    for word in sent:
      vocab.add(word[0].lower())
      label.add(word[1])
  vocab = list(vocab)
  label = list(label)
  #vocab_dict = {x:vocab.index(x) for x in vocab}
  return vocab, label

def word2features(sent, i, vocab, window_size=7):
  unk_index = vocab.index("<UNK>")
  features_word = []
  features_addition = []
  for j in range(-window_size, window_size+1):
    if i + j < 0 or i + j >= len(sent):
      features_word.append(unk_index)
      features_addition.append([0,0,0])
    else:
      features_word.append(vocab.index(sent[i][0].lower()))
      features_addition.append([
        int(sent[i][0].islower()),
        int(sent[i][0].istitle()),
        int(sent[i][0].isdigit())
      ])
  return features_word, features_addition

def sent2features(sent, vocab, window_size=7):
  features_word = []
  features_addition = []
  for i in range(len(sent)):
    fw, fa = word2features(sent, i, vocab, window_size)
    features_word.append(fw)
    features_addition.append(fa)
  return features_word, features_addition

def sent2label(sent, label_vocab):
  return [label_vocab.index(sent[i][1]) for i in range(len(sent))]

def data2window(file_path, word_vocab, label_vocab, window_size=7):
  sentences = corpus_reader(file_path)
  windows_word , window_addition, windows_label = [], [], []
  for sent in sentences:
    tw, ta = sent2features(sent, word_vocab, window_size)
    tl = sent2label(sent, label_vocab)
    windows_word.extend(tw)
    window_addition.extend(ta)
    windows_label.extend(tl)
  return windows_word, window_addition, windows_label
                      
def get_full_vocab(file_paths):
  full_word_vocab = set()
  full_label_vocab = set()
  for file in file_paths:
    twv, tlv = vocab_encode(corpus_reader(file))
    full_word_vocab = full_word_vocab.union(twv)
    full_label_vocab = full_label_vocab.union(tlv)
  return list(full_word_vocab), list(full_label_vocab)
