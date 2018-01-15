from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import sklearn
import pycrfsuite
from crf_pos import corpus_reader, sent2labels, sent2features, sent2tokens
from itertools import chain

def pos_classification_report(y_true, y_pred):
  """
  Classification report for a list of BIO-encoded sequences.
  It computes token-level metrics and discards "O" labels.

  Note that it requires scikit-learn 0.15+ (or a version from github master)
  to calculate averages properly!
  """
  lb = LabelBinarizer()
  y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
  y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))
        
  tagset = set(lb.classes_)
  tagset = sorted(tagset)
  class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}
    
  return classification_report(
    y_true_combined,
    y_pred_combined,
    labels = [class_indices[cls] for cls in tagset],
    target_names = tagset,
  )

if __name__ == "__main__":
  tagger = pycrfsuite.Tagger()
  tagger.open('BKTreeBank_LREC2018.crfsuite')
  
  test_sents = corpus_reader("../BKTreebank_LREC2018/test03_cleaned")
  
  X_test = [sent2features(sent) for sent in test_sents]
  y_test = [sent2labels(sent) for sent in test_sents]

  #example_sent = test_sents[1]
  #print(' '.join(sent2tokens(example_sent)))
  
  #print("Predicted:", ' '.join(tagger.tag(sent2features(example_sent))))
  #print("Correct:  ", ' '.join(sent2labels(example_sent)))
  
  y_pred = [tagger.tag(x) for x in X_test]
  
  print(pos_classification_report(y_test, y_pred))
