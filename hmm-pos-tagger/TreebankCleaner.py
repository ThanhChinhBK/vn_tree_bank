######### TreebankCleaner.py #########

from Helper import msg # for messaging
import os

def read_conll_sequence_labeling(path, index_word, index_label):
  """
  read data from file in conll format
  :param path: file path
  :return: sentences of words and labels, sentences of indexes of words and labels.
  """
  word_sentences = []
  label_sentences = []
  words = []
  labels = []
  num_tokens = 0
  MAX_LENGTH = 120

  with open(path, "rb") as file:
    for line in file:
      line = line.decode('utf-8')
      if line.strip() == "":#this means we have the entire sentence
        if 0 < len(words) <= MAX_LENGTH:
          word_sentences.append(words[:])
          label_sentences.append(labels[:])
          num_tokens += len(words)

        else:
          if len(words) != 0:
            print("ignore sentence with length %d" % (len(words)))

        words = []
        labels = []
      else:
        tokens = line.strip().split()
        word = tokens[index_word]
        label = tokens[index_label]
        words.append(word)
        #if(out_dir !=None):
        #  vocab.add(word)
        labels.append(label)
 
   #this is for the last sentence            
    if 0 < len(words) <= MAX_LENGTH:
      word_sentences.append(words[:])
      label_sentences.append(labels[:])
      num_tokens += len(words)
    else:
      if len(words) != 0:
        print("ignore sentence with length %d" % (len(words)))

 
  print("#sentences: %d, #tokens: %d" % (len(word_sentences), num_tokens))
    
  return word_sentences, label_sentences

class TreebankCleaner:
  "A class for cleaning treebank text"
    
  def __init__(self, corpus_path, corpus_files):
    """
    Initialize a TreebankCleaner object.
    
    :param corpus_path: path of corpus files
    :param corpus_files: list of corpus files
    """
        
    self.corpus_path = corpus_path
    self.corpus_files = corpus_files
    
    ######### `PUBLIC' FUNCTIONS #########

        
  def clean(self):
    """
        Clean corpus files and write the results to disk
        """
        
    # loop through files
    for corpus_file in self.corpus_files:
            
      msg("Cleaning %s..." % corpus_file)
      word_sents, label_sents = read_conll_sequence_labeling(os.path.join(self.corpus_path, corpus_file),1,3)
      with open(os.path.join(self.corpus_path, corpus_file+"_cleaned"), "w") as f:
        for word, label in zip(word_sents, label_sents):
          for w, l in zip(word, label):
            f.write(w+"/"+l+" ")
          f.write("\n")
    
