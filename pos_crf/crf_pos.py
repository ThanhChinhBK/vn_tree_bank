import nltk 
import os
import numpy
import pycrfsuite

def corpus_reader(file_path):
  sentences = []
  with open(file_path) as f:
    for line in f:
      sentences.append([nltk.tag.str2tuple(token) for token in line.strip().split(" ")])
  return sentences

def word2features(sent, i):
  word = sent[i][0]
  features = [
    'bias',
    'word.lower=' + word.lower(),
    'word[-3:]=' + word[-3:],
    'word[-2:]=' + word[-2:],
    'word.isupper=%s' % word.isupper(),
    'word.istitle=%s' % word.istitle(),
    'word.isdigit=%s' % word.isdigit(),
  ]
  if i > 0:
    word1 = sent[i-1][0]
    postag1 = sent[i-1][1]
    features.extend([
      '-1:word.lower=' + word1.lower(),
      '-1:word.istitle=%s' % word1.istitle(),
      '-1:word.isupper=%s' % word1.isupper(),
    ])
  else:
    features.append('BOS')
    
  if i < len(sent)-1:
    word1 = sent[i+1][0]
    postag1 = sent[i+1][1]
    features.extend([
      '+1:word.lower=' + word1.lower(),
      '+1:word.istitle=%s' % word1.istitle(),
      '+1:word.isupper=%s' % word1.isupper(),
    ])
  else:
    features.append('EOS')
    
  return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, label in sent]

def sent2tokens(sent):
    return [token for token,label in sent]    

if __name__ == "__main__":
  train_sents = corpus_reader("../BKTreebank_LREC2018/train03-train_cleaned")
  dev_sents = corpus_reader("../BKTreebank_LREC2018/train03-dev_cleaned")
  test_sents = corpus_reader("../BKTreebank_LREC2018/test03_cleaned")
  train_sents = train_sents + dev_sents

  X_train = [sent2features(sent) for sent in train_sents]
  y_train = [sent2labels(sent) for sent in train_sents]
  
  X_test = [sent2features(sent) for sent in test_sents]
  y_test = [sent2labels(sent) for sent in test_sents]

  trainer = pycrfsuite.Trainer(verbose=False)

  for xseq, yseq in zip(X_train, y_train):
    trainer.append(xseq, yseq)
  
  trainer.set_params({
    'c1':1.0,
    'c2':1e-3,
    'max_iterations':1000000,
    'feature.possible_transitions':True
  })

  print(trainer.params)

  trainer.train('BKTreeBank_LREC2018.crfsuite')

  print(trainer.logparser.last_iteration)
    
