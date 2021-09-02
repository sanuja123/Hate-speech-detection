from string import punctuation
from os import listdir
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

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
	tokens = [w.translate(table) for w in tokens]
	# filter out tokens not in vocab
	tokens = [w for w in tokens if w in vocab]
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
			#continue
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
ytrain = array([0 for _ in range(2790)] + [1 for _ in range(2790)])

# load all test reviews
positive_docs = process_docs('D:/Research/implementation/new data for cnn/implementing_data/test_pos', vocab, False)
negative_docs = process_docs('D:/Research/implementation/new data for cnn/implementing_data/test_neg', vocab, False)
test_docs = negative_docs + positive_docs
# sequence encode
encoded_docs = tokenizer.texts_to_sequences(test_docs)
# pad sequences
Xtest = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
# define test labels
ytest = array([0 for _ in range(100)] + [1 for _ in range(100)])

# define vocabulary size (largest integer value)
vocab_size = len(tokenizer.word_index) + 1

# define model
model = Sequential()
model.add(Embedding(vocab_size, 100, input_length=max_length))
model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
print(model.summary())
# compile network
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit network
model.fit(Xtrain, ytrain, epochs=20, verbose=2)
# evaluate
loss, acc = model.evaluate(Xtest, ytest, verbose=0)
print('Test Accuracy: %f' % (acc*100))