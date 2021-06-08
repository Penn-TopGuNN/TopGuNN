
import string
import re
import collections
from collections import defaultdict, Counter
import json
import math
import util
import gzip
import numpy as np
from tqdm import tqdm 
import util
import random

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('treebank')
nltk.data.load('tokenizers/punkt/english.pickle')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english')) 
from nltk.tokenize import word_tokenize, sent_tokenize, TweetTokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import linalg
from sklearn.preprocessing import normalize,MinMaxScaler,StandardScaler

def get_row_vector(matrix, row_id):
  return matrix[row_id, :]

def get_column_vector(matrix, col_id):
  return matrix[:, col_id]

def create_term_document_matrix(line_tuples, document_names, vocab):
  '''Returns a numpy array containing the term document matrix for the input lines.

  Inputs:
    line_tuples: A list of tuples, containing the name of the document and a tokenized line from that document.
    document_names: A list of the document names
    vocab: A list of the tokens in the vocabulary (only extracting verb-events from the vocabulary)

  # NOTE: THIS DOCSTRING WAS UPDATED ON JAN 24, 12:39 PM.

  Let m = len(vocab) and n = len(document_names).

  Returns:
    td_matrix: A mxn numpy array where the number of rows is the number of words
        and each column corresponds to a document. A_ij contains the
        frequency with which word i occurs in document j.
  '''
  # YOUR CODE HERE
  vocab_to_id = dict(zip(vocab, range(0, len(vocab))))
  docname_to_id = dict(zip(document_names, range(0, len(document_names))))
  nrows, ncols = (len(vocab), len(document_names))

  td_matrix = np.zeros((nrows,ncols))

  for name,line in line_tuples:
    doc_name, tokens = (name, line)
    doc_id = docname_to_id[doc_name] #33
    for word in tokens:
      try: 
        word_id = vocab_to_id[word] #3568
        td_matrix[word_id,doc_id] += 1
      except:
        print('exception error!!!')
        print(word)
        print(tokens)
        continue

  return td_matrix

def create_tf_idf_matrix(term_document_matrix):
  '''Given the term document matrix, output a tf-idf weighted version.

  See section 15.2.1 in the textbook.
  
  Hint: Use numpy matrix and vector operations to speed up implementation.

  Input:
    term_document_matrix: Numpy array where each column represents a document 
    and each row, the frequency of a word in that document.

  Returns:
    A numpy array with the same dimension as term_document_matrix, where
    A_ij is weighted by the inverse document frequency of document h.
  '''

  # YOUR (CODE HERE
  x, N = term_document_matrix.shape
  df = np.sum(np.count_nonzero(term_document_matrix, axis=1))
  idf = np.log(float(N)/df)
  tf_idf_matrix = np.multiply(term_document_matrix,idf)
  
  return tf_idf_matrix

def rank_queries(target_play_index, term_document_matrix, similarity_fn):
  ''' Ranks the similarity of all of the plays to the target play.

  # NOTE: THIS DOCSTRING WAS UPDATED ON JAN 24, 12:51 PM.

  Inputs:
    target_play_index: The integer index of the play we want to compare all others against.
    term_document_matrix: The term-document matrix as a mxn numpy array.
    similarity_fn: Function that should be used to compared vectors for two
      documents. Either compute_dice_similarity, compute_jaccard_similarity, or
      compute_cosine_similarity.

  Returns:
    A length-n list of integer indices corresponding to play names,
    ordered by decreasing similarity to the play indexed by target_play_index

  def get_row_vector(matrix, row_id):
    return matrix[row_id, :]

  def get_column_vector(matrix, col_id):
    return matrix[:, col_id]
  '''
  
  # YOUR CODE HERE
  verbs, queries = term_document_matrix.shape
  similarity_results = np.zeros(queries)

  target = get_column_vector(term_document_matrix, target_play_index)

  for i in range(queries):
    curr_vec = get_column_vector(term_document_matrix,i)
    curr_sim = similarity_fn(curr_vec.T, target.T) ## transpose column vecs
    np.put(similarity_results,[i],curr_sim)

  sorted_indexes = np.argsort(-similarity_results)
  sorted_vals = -np.sort(-similarity_results)

  return [x for x in sorted_indexes], [x for x in sorted_vals]

def compute_cosine_similarity(vector1, vector2):
  '''Computes the cosine similarity of the two input vectors.

  Inputs:
    vector1: A nx1 numpy array
    vector2: A nx1 numpy array

  Returns:
    A scalar similarity value.
  '''
  
  # YOUR CODE HERE
  vw = np.dot(vector1,vector2.T)
  unit_v = np.sqrt(np.sum(np.square(vector1))) 
  unit_w = np.sqrt(np.sum(np.square(vector2)))

  return float(vw)/(float(unit_v*unit_w) + 1e-10)

def compute_jaccard_similarity(vector1, vector2):
  '''Computes the cosine similarity of the two input vectors.

  Inputs:
    vector1: A nx1 numpy array
    vector2: A nx1 numpy array

  Returns:
    A scalar similarity value.
  '''
  
  # YOUR CODE HERE
  intersect = np.sum(np.minimum(vector1,vector2))
  union = np.sum(np.maximum(vector1,vector2))

  return float(intersect)/(float(union) + 1e-10)

def compute_dice_similarity(vector1, vector2):
  '''Computes the cosine similarity of the two input vectors.

  Inputs:
    vector1: A nx1 numpy array
    vector2: A nx1 numpy array

  Returns:
    A scalar similarity value.
  '''
  # YOUR CODE HERE
  numerator = 2.0*compute_jaccard_similarity(vector1,vector2)
  denominator = compute_jaccard_similarity(vector1,vector2) + 1

  return float(numerator) / (float(denominator) + 1e-10)

def get_events(sentence):
  verb_tags = set(['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'])
  verbs =[]
  tokenized_sentence = word_tokenize(sentence)
  tagged = nltk.pos_tag(tokenized_sentence)
  words = [x.lower() for (x,y) in tagged]
  pos_tags = [y for (x,y) in tagged]
  for i, word in enumerate(words):
    if pos_tags[i] in verb_tags:
      verbs.append(word)
  return verbs

# nlp_analysis.display_analysis(line_tuples, document_names, list(set(vocab)), query_num)
def display_analysis(line_tuples, document_names, vocab, query_num):

  td_matrix = create_term_document_matrix(line_tuples, document_names, vocab)
  tf_idf_matrix = create_tf_idf_matrix(td_matrix)

  print("\nterm document: ")
  random_idx = random.randint(0, len(document_names)-1)
  similarity_fns = [compute_cosine_similarity, compute_jaccard_similarity, compute_dice_similarity]
  for sim_fn in similarity_fns:
      results=[]
      print('\nThe 10 most similar queries to "%s" using %s are:' % (document_names[random_idx], sim_fn.__qualname__))
      ranks, sims = rank_queries(random_idx, td_matrix, sim_fn)
      for idx in range(0, 10):
        doc_id = ranks[idx]
        print('%d: %s %.3f' % (idx+1, document_names[doc_id], sims[idx]))
        results.append('%d: %s %.3f' % (idx+1, document_names[doc_id], sims[idx]))
      util.write_outfile(results, "out/td_matrix_"+str(query_num)+"_"+str(sim_fn)[7:]+".txt")

  print("\ntf-idf matrix: ")
  random_idx = random.randint(0, len(document_names)-1)
  similarity_fns = [compute_cosine_similarity, compute_jaccard_similarity, compute_dice_similarity]
  for sim_fn in similarity_fns:
      results=[]
      print('\nThe 10 most similar queries to "%s" using %s are:' % (document_names[random_idx], sim_fn.__qualname__))
      ranks, sims = rank_queries(random_idx, tf_idf_matrix, sim_fn)
      for idx in range(0, 10):
        doc_id = ranks[idx]
        print('%d: %s %.3f' % (idx+1, document_names[doc_id], sims[idx]))
        results.append('%d: %s %.3f' % (idx+1, document_names[doc_id], sims[idx]))
      util.write_outfile(results, "out/tf-idf_matrix_"+str(query_num)+"_"+str(sim_fn)[7:]+".txt")

        # curr_vocab = nlp_analysis.get_events(q_sentences[query_num])
        # line_tuples.append(('Q_Text'+str(query_num), curr_vocab))
        # document_names.append('Q_Text'+str(query_num))
        # vocab.extend(curr_vocab)
            # curr_vocab = nlp_analysis.get_events(nn_sentence) 
            # line_tuples.append(('KNN'+str(match_num), curr_vocab))
            # document_names.append('KNN'+str(match_num))
            # vocab.extend(curr_vocab)

# from sklearn.metrics.pairwise import cosine_similarity
# ## Calculates similarity between 2 vectors

def calc_similarity(x, y):
    nx = np.asarray(x).reshape(1, -1)
    ny = np.asarray(y).reshape(1, -1)
    return cosine_similarity(nx, ny)[0][0]

def cosine_sim(vector1, vector2):
  '''Computes the cosine similarity of the two input vectors.

  Inputs:
    vector1: A nx1 numpy array
    vector2: A nx1 numpy array

  Returns:
    A scalar similarity value.
  '''
  
  # YOUR CODE HERE
  cos=np.dot(vector1, vector2)/(np.linalg.norm(vector1)*np.linalg.norm(vector2))
  return cos

# def cosine_sim(vector1, vector2):
#   '''Computes the cosine similarity of the two input vectors.

#   Inputs:
#     vector1: A nx1 numpy array
#     vector2: A nx1 numpy array

#   Returns:
#     A scalar similarity value.
#   '''
  
#   # YOUR CODE HERE
#   vw = np.dot(vector1,vector2.T)
#   unit_v = np.sqrt(np.sum(np.square(vector1))) 
#   unit_w = np.sqrt(np.sum(np.square(vector2)))

#   return float(vw)/(float(unit_v*unit_w) + 1e-10)