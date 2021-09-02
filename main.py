from string import punctuation
from os import listdir
from numpy import array
from numpy import asarray
from numpy import zeros
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

from keras.layers.advanced_activations import LeakyReLU

import numpy as np
#from glove import Corpus
# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, encoding="utf8")
	#file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# turn a doc into clean tokens
def clean_doc(doc, vocab):
	# split into tokens by white space
	tokens = doc.split()
	# remove punctuation from each token
	table = str.maketrans('', '', punctuation)
	#tokens = [w.translate(table) for w in tokens]
	# filter out tokens not in vocab
	#tokens = [w for w in tokens if w in vocab]
	tokens = ' '.join(tokens)
	return tokens

# load all docs in a directory
def process_docs(directory, vocab, is_trian):
	documents = list()
	# walk through all files in the folder
	for filename in listdir(directory):
		# skip any reviews in the test set
		#if is_trian and filename.startswith('cv9'):
			#continue
		#if not is_trian and not filename.startswith('cv9'):
			# continue
		# create the full path of the file to open
		path = directory + '/' + filename
		# load the doc

	#file = open(filename, encoding="utf8")

		file1 = open(path, 'r',encoding="utf8")
		Lines = file1.readlines()

		# Strips the newline character
		for line in Lines:
			myText = open(r'D:/Research/implementation/new data for cnn/implementing_data/test_text/my_text_file.txt', 'w',encoding="utf8")
			myString = line
			myText.write(myString)
			myText.close()

			doc = load_doc('D:/Research/implementation/new data for cnn/implementing_data/test_text/my_text_file.txt')
			# clean doc
			tokens = clean_doc(doc, vocab)
			# add to list
			documents.append(tokens)


	return documents

# load embedding as a dict
def load_embedding(filename):
	# load embedding into memory, skip first line
	file = open(filename, encoding="utf8")
	lines = file.readlines()[1:]
	file.close()
	# create a map of words to vectors
	embedding = dict()
	for line in lines:
		parts = line.split()
		# key is string word, value is numpy array for vector
		embedding[parts[0]] = asarray(parts[1:], dtype='float32')
	return embedding

# create a weight matrix for the Embedding layer from a loaded embedding
def get_weight_matrix(embedding, vocab):
	# total vocabulary size plus 0 for unknown words
	vocab_size = len(vocab) + 1
	# define weight matrix dimensions with all 0
	weight_matrix = zeros((vocab_size, 100))
	# step vocab, store vectors using the Tokenizer's integer mapping
	for word, i in vocab.items():
		weight_matrix[i-1] = embedding.get(word)
	return weight_matrix

# load the vocabulary
vocab_filename = 'D:/Research/implementation/new data for cnn/implementing_data/vocab/vocab.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)

# load all training reviews
positive_docs = process_docs('D:/Research/implementation/new data for cnn/implementing_data/train_pos', vocab, True)
negative_docs = process_docs('D:/Research/implementation/new data for cnn/implementing_data/train_neg', vocab, True)
train_docs = negative_docs + positive_docs

# create the tokenizer
tokenizer = Tokenizer()
# fit the tokenizer on the documents
tokenizer.fit_on_texts(train_docs)

# sequence encode
encoded_docs = tokenizer.texts_to_sequences(train_docs)
# pad sequences
max_length = max([len(s.split()) for s in train_docs])
Xtrain = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
# define training labels
#ytrain = array([0 for _ in range(1)] + [1 for _ in range(1)])

length = len(Xtrain)
middle_index = length//2

ytrain = array([0 for _ in range(2790)] + [1 for _ in range(2790)])
# load all test reviews
positive_docs = process_docs('D:/Research/implementation/new data for cnn/implementing_data/test_pos', vocab, False)
negative_docs = process_docs('D:/Research/implementation/new data for cnn/implementing_data/test_neg', vocab, False)
test_docs = negative_docs + positive_docs
# sequence encode
encoded_docs = tokenizer.texts_to_sequences( test_docs)
# pad sequences
Xtest = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
# define test labels
ytest = array([0 for _ in range(100)] + [1 for _ in range(100)])

# define vocabulary size (largest integer value)
vocab_size = len(tokenizer.word_index) + 1

# load embedding from file

#D:/Research/implementation/implementing_data/all/embedding_word2vec.txt



#raw_embedding = load_embedding("D:/Research/implementation/new data for cnn/word2vec/embedding_word2vec.txt")
# get vectors in the right order
#embedding_vectors = get_weight_matrix(raw_embedding, tokenizer.word_index)
# create the embedding layer
#embedding_layer = Embedding(vocab_size, 100, weights=[embedding_vectors], input_length=max_length, trainable=True )



#NEw ebedding

embeddings_index = {}
#f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))

file = open("D:/Research/implementation/new data for cnn/word2vec/embedding_word2vec.txt", encoding="utf8")
for line in file:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
file.close()

print('Found %s word vectors.' % len(embeddings_index))
embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, 100))
for word, i in tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(len(tokenizer.word_index) + 1,
                            100,
                            weights=[embedding_matrix],
                            input_length=max_length,
                            trainable=True)
#end new embedding


# define model
model = Sequential()
model.add(embedding_layer)


model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))

model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))

model.add(MaxPooling1D(pool_size=2))

model.add(Flatten())
model.add(Dense(1, activation='relu'))
print(model.summary())

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit network
model.fit(Xtrain, ytrain, epochs=20, verbose=2)
# evaluate
loss, acc = model.evaluate(Xtest, ytest, verbose=0)
print('Test Accuracy: %f' % (acc * 100))