import numpy as np
import pickle
import scipy.io

## Import data

# Choose the dataset
path = 'data/min_df_100'
#path = 'data/min_df_30'

# Read in
counts_training_mat = scipy.io.loadmat(path+'/bow_tr_counts.mat')
counts_training = counts_training_mat['counts']
tokens_training_mat = scipy.io.loadmat(path+'/bow_tr_tokens.mat')
tokens_training = tokens_training_mat['tokens']

counts_validation_mat = scipy.io.loadmat(path+'/bow_va_counts.mat')
counts_validation = counts_validation_mat['counts']
tokens_validation_mat = scipy.io.loadmat(path+'/bow_va_tokens.mat')
tokens_validation = tokens_validation_mat['tokens']

counts_test1_mat = scipy.io.loadmat(path+'/bow_ts_h1_counts.mat')
counts_test1 = counts_test1_mat['counts']
tokens_test1_mat = scipy.io.loadmat(path+'/bow_ts_h1_tokens.mat')
tokens_test1 = tokens_test1_mat['tokens']

counts_test2_mat = scipy.io.loadmat(path+'/bow_ts_h2_counts.mat')
counts_test2 = counts_test2_mat['counts']
tokens_test2_mat = scipy.io.loadmat(path+'/bow_ts_h2_tokens.mat')
tokens_test2 = tokens_test2_mat['tokens']

with open(path + '/vocab.pkl', 'rb') as file:
    vocab = pickle.load(file)

vocab_size = len(vocab)

def create_document_term_matrix(tokens, counts, vocab_size):
  n_docs = tokens.shape[1]
  document_term_matrix = np.zeros((n_docs,vocab_size))
  for document in range(n_docs):
    words_document = tokens[0,document][0,:]
    counts_document = counts[0,document][0,:]
    for idx_word, word in enumerate(words_document):
      document_term_matrix[document,word]=counts_document[idx_word]
  return document_term_matrix

training_set = create_document_term_matrix(tokens_training,counts_training,vocab_size)
valid = create_document_term_matrix(tokens_validation,counts_validation,vocab_size)
test_1 = create_document_term_matrix(tokens_test1,counts_test1,vocab_size)
test_2 = create_document_term_matrix(tokens_test2,counts_test2,vocab_size)


## Get data sizes

# Training data
num_of_docs_train = training_set.shape[0]

# Validation set
num_of_docs_valid = valid.shape[0]

# Test data
num_of_docs_test = test_1.shape[0] + test_2.shape[0]

num_of_docs_test_1 = test_1.shape[0]
num_of_docs_test_2 = test_2.shape[0]