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
import matplotlib.pyplot as plt
from keras.layers.advanced_activations import LeakyReLU
import keras
import numpy as np
from keras.optimizers import SGD
from keras.models import Model





from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from keras.layers import Concatenate, Dense, LSTM, Input, concatenate
from keras.layers import merge
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

ytrain = array([0 for _ in range(2790)] + [1 for _ in range(3355)])
# load all test reviews
positive_docs = process_docs('D:/Research/implementation/new data for cnn/implementing_data/test_pos', vocab, False)
negative_docs = process_docs('D:/Research/implementation/new data for cnn/implementing_data/test_neg', vocab, False)
test_docs = negative_docs + positive_docs
# sequence encode
encoded_docs = tokenizer.texts_to_sequences( test_docs)
# pad sequences
Xtest = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

print(max_length)

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

file = open("D:/Research/implementation/implementing_data/all/embedding_word2vec.txt", encoding="utf8")
for line in file:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
file.close()

print('Found %s word vectors.' % len(embeddings_index))
embedding_matrix1 = np.zeros((len(tokenizer.word_index) + 1, 100))
for word, i in tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix1[i] = embedding_vector

embedding_layer1 = Embedding(len(tokenizer.word_index) + 1,
                            100,
                            weights=[embedding_matrix1],
                            input_length=max_length,
                            trainable=False)



embeddings_index = {}
#f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))

file = open("D:/Research/implementation/implementing_data/all/glove.txt", encoding="utf8")
for line in file:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
file.close()

print('Found %s word vectors.' % len(embeddings_index))
embedding_matrix2 = np.zeros((len(tokenizer.word_index) + 1, 100))
for word, i in tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix2[i] = embedding_vector

embedding_layer2 = Embedding(len(tokenizer.word_index) + 1,
                            100,
                            weights=[embedding_matrix2],
                            input_length=max_length,
                            trainable=True)



#end new embedding


inputs = keras.layers.Input(((166,) ), name="input", )
print(type(inputs))

embedding1 =Embedding(vocab_size, 100, weights=[embedding_matrix1], input_length=4, trainable=False)(inputs)
conv_w2v1 = Conv1D(32, kernel_size=8, activation='relu')(embedding1)
print(conv_w2v1)
pool_w2v1 = MaxPooling1D(pool_size=(2))(conv_w2v1)

conv_w2v2 = Conv1D(32, kernel_size=8, activation='relu')(pool_w2v1)
pool_w2v2 = MaxPooling1D(pool_size=(2))(conv_w2v2)

flat_w2v = Flatten()(pool_w2v2)


embedding2 =Embedding(vocab_size, 100, weights=[embedding_matrix2], input_length=4, trainable=False)(inputs)
conv_glove1 = Conv1D(32, kernel_size=8, activation='relu')(embedding2)
pool_glove1 = MaxPooling1D(pool_size=(2))(conv_glove1)

conv_glove2 = Conv1D(32, kernel_size=8, activation='relu')(pool_glove1)
pool_glove2 = MaxPooling1D(pool_size=(2))(conv_glove2)

flat_glove = Flatten()(pool_glove2)


merge_flat = concatenate([flat_w2v, flat_glove])

hidden0 = Dense(1024, activation='relu')(merge_flat)
droout0 = Dropout(0.3)(hidden0)

hidden1 = Dense(256, activation='relu')(droout0)
droout1 = Dropout(0.3)(hidden1)

hidden2 = Dense(256, activation='relu')(droout1)
droout2 = Dropout(0.3)(hidden2)

hidden3 = Dense(128, activation='relu')(droout2)
droout3 = Dropout(0.3)(hidden3)

hidden4 = Dense(64, activation='relu')(droout3)
droout4 = Dropout(0.3)(hidden4)

output = Dense(1, activation='sigmoid')(droout4)
model = Model(inputs=inputs, outputs=output)
# summarize layers
print(model.summary())


opt = SGD(lr=0.001, momentum=0.9)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit network
history = model.fit(Xtrain, ytrain, validation_data=(Xtest, ytest), epochs=15,shuffle=True, verbose=2)
#history = model.fit(Xtrain, ytrain,epochs=20,shuffle=True, verbose=2,validation_split=0.3)
# evaluate
_, train_acc = model.evaluate(Xtrain, ytrain, verbose=0)
_, test_acc = model.evaluate(Xtest, ytest, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

loss, accuracy = model.evaluate(Xtest, ytest, verbose=0)
print('Accuracy: %f' % (accuracy*100))

#yhat_probs=model.predict(Xtest)
#yhat_classes=np.argmax(yhat_probs,axis=-1)
# predict probabilities for test set
yhat_probs = model.predict(Xtest, verbose=0)
# predict crisp classes for test set
yhat_classes  = (yhat_probs > 0.5).astype(int)
# reduce to 1d array
yhat_probs = yhat_probs[:, 0]
yhat_classes = yhat_classes[:, 0]
import tensorflow
#y_classes = tensorflow.keras.utils.np_utils.probas_to_classes(model.predict(Xtest))
# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(ytest, yhat_classes)

print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(ytest, yhat_classes)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(ytest, yhat_classes)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(ytest, yhat_classes)
print('F1 score: %f' % f1)




plt.figure(1)

# summarize history for accuracy

plt.figure(1)

# summarize history for accuracy

plt.subplot(211)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='lower right')

# summarize history for loss

plt.subplot(212)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper right')

plt.tight_layout()

plt.show()