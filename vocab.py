from string import punctuation
from os import listdir
from collections import Counter
from nltk.corpus import stopwords

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, encoding="utf8")
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# turn a doc into clean tokens
def clean_doc(doc):
	myText = open(r'D:/Research/implementation/new data for cnn/implementing_data/vocab_test.txt', 'w',encoding="utf8")
	# split into tokens by white space
	text = doc.split()
	tokens = doc.split()
	# remove punctuation from each token
	#table = str.maketrans('', '', punctuation)
	#tokens = [w.translate(table) for w in tokens]
	# remove remaining tokens that are not alphabetic
	#tokens = [word for word in tokens if word.isalpha()]
	# filter out stop words
	#stop_words = set(stopwords.words('english'))
	#tokens = [w for w in tokens if not w in stop_words]
	# filter out short tokens
	tokens = [word for word in tokens if len(word) > 1]
	for element in tokens:
		myText.write(element + "\n")
	myText.close()
	#myText.write(text)
	return tokens

# load doc and add to vocab
def add_doc_to_vocab(filename, vocab):
	# load doc
	doc = load_doc(filename)
	# clean doc
	tokens = clean_doc(doc)
	# update counts
	vocab.update(tokens)

# load all docs in a directory
def process_docs(directory, vocab, is_trian):
	# walk through all files in the folder
	for filename in listdir(directory):
		# skip any reviews in the test set
		if is_trian and filename.startswith('cv9'):
			continue
		if not is_trian and not filename.startswith('cv9'):
			continue
		# create the full path of the file to open
		path = directory + '/' + filename
		# add doc to vocab
		add_doc_to_vocab(path, vocab)

# define vocab
vocab = Counter()
# add all docs to vocab
process_docs('D:/Research/implementation/implementing_data/positive', vocab, True)
process_docs('D:/Research/implementation/implementing_data/negative', vocab, True)

# print the size of the vocab
print(len(vocab))
# print the top words in the vocab
print(vocab.most_common(50))