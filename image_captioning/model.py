from pickle import load
import os
os.chdir('C:\\Users\\chrisb\\OneDrive - Leesa\\jobs\\developingInPython\\image_captioning')
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, Add
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import json
 
# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text
 
# load a pre-defined list of photo identifiers
def load_set(filename):
	doc = load_doc(filename)
	dataset = list()
	# process line by line
	for line in doc.split('\n'):
		# skip empty lines
		if len(line) < 1:
			continue
		# get the image identifier
		identifier = line.split('.')[0]
		dataset.append(identifier)
	return set(dataset)
 
# load clean descriptions into memory
def load_clean_descriptions(filename, dataset):
	# load document
	doc = load_doc(filename)
	descriptions = dict()
	for line in doc.split('\n'):
		# split line by white space
		tokens = line.split()
		# split id from description
		image_id, image_desc = tokens[0], tokens[1:]
		# skip images not in the set
		if image_id in dataset:
			# create list
			if image_id not in descriptions:
				descriptions[image_id] = list()
			# wrap description in tokens
			desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
			# store
			descriptions[image_id].append(desc)
	return descriptions
 
# load photo features
def load_photo_features(filename, dataset):
	# load all features
	all_features = load(open(filename, 'rb'))
	# filter features
	features = {k: all_features[k] for k in dataset}
	return features

# convert a dictionary of clean descriptions to a list of descriptions
def to_lines(descriptions):
	all_desc = list()
	for key in descriptions.keys():
		[all_desc.append(d) for d in descriptions[key]]
	return all_desc
 
# fit a tokenizer given caption descriptions
def create_tokenizer(descriptions):
	lines = to_lines(descriptions)
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer

# calculate the length of the description with the most words
def max_length(descriptions):
	lines = to_lines(descriptions)
	return max(len(d.split()) for d in lines)



#altered for generator
def create_sequences(tokenizer, max_length, desc_list, photo):
	X1, X2, y = list(), list(), list()
	# walk through each description for the image
	for desc in desc_list:
		# encode the sequence
		seq = tokenizer.texts_to_sequences([desc])[0]
		# split one sequence into multiple X,y pairs
		for i in range(1, len(seq)):
			# split into input and output pair
			in_seq, out_seq = seq[:i], seq[i]
			# pad input sequence
			in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
			# encode output sequence
			out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
			# store
			X1.append(photo)
			X2.append(in_seq)
			y.append(out_seq)
	return np.array(X1), np.array(X2), np.array(y)

# create sequences of images, input sequences and output words for an image
def create_sequences(tokenizer, max_len, descriptions, photos):
	X1, X2, y = list(), list(), list()
	# walk through each image identifier
	for key, desc_list in descriptions.items():
		#key = '17273391_55cfc7d3d4'
		#desc_list = descriptions['17273391_55cfc7d3d4']
		# walk through each description for the image
		for desc in desc_list:
			# encode the sequence
			"""this transforms words to index, keeps in list"""
			seq = tokenizer.texts_to_sequences([desc])[0]
			# split one sequence into multiple X,y pairs
			for i in range(1, len(seq)):
				# split into input and output pair
				"""get cumulative list up to point for next word"""
				in_seq, out_seq = seq[:i], seq[i]
				# pad input sequence
				"""every in_seq is same length -- which is equal to max len"""
				in_seq = pad_sequences([in_seq], maxlen=max_len)[0]
				# encode output sequence
				"""every outword is is 1 of total vocab"""
				out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
				# store
				X1.append(photos[key][0])
				X2.append(in_seq)
				y.append(out_seq)
	"""every output is num_photos by num of words in desc by num of descriptions"""
	return np.array(X1), np.array(X2), np.array(y)

# define the captioning model
def define_model(vocab_size, max_length):
	# feature extractor model
	inputs1 = Input(shape=(4096,))
	fe1 = Dropout(0.5)(inputs1)
	fe2 = Dense(256, activation='relu')(fe1)
	# sequence model
	inputs2 = Input(shape=(max_length,))
	se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
	se2 = Dropout(0.5)(se1)
	se3 = LSTM(256)(se2)
	# decoder model
	decoder1 = Add()([fe2, se3])
	decoder2 = Dense(256, activation='relu')(decoder1)
	outputs = Dense(vocab_size, activation='softmax')(decoder2)
	# tie it together [image, seq] [word]
	model = Model(inputs=[inputs1, inputs2], outputs=outputs)
	model.compile(loss='categorical_crossentropy', optimizer='adam')
	# summarize model
	print(model.summary())
	plot_model(model, to_file='model.png', show_shapes=True)
	return model


"""load training dataset (6K) - SET of 6000 images IDS"""
train_image_data = 'data/raw_text/Flickr_8k.trainImages.txt'
train = load_set(train_image_data)
print('Dataset: %d' % len(train))
"""descriptions-- DICT - KEY is image ID, VALUE is list of full description of same image"""
train_descriptions = load_clean_descriptions('data/descriptions.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))
"""photo features -- DICT -- KEY is name, value is compressed vector of 1*1000"""
train_features = load_photo_features('data/features.pkl', train)
print('Photos: train=%d' % len(train_features))
# prepare tokenizer
"""gets all descriptoins in single list then pass through tokenizer"""
tokenizer = create_tokenizer(train_descriptions)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)
# load test set
test_image_data = 'data/raw_text/Flickr_8k.devImages.txt'
test = load_set(test_image_data)
print('Dataset: %d' % len(test))
# descriptions
test_descriptions = load_clean_descriptions('data/descriptions.txt', test)
print('Descriptions: test=%d' % len(test_descriptions))
# photo features
test_features = load_photo_features('data/features.pkl', test)
print('Photos: test=%d' % len(test_features))
# prepare sequences
# determine the maximum sequence length
max_len = max_length(train_descriptions) #make lenght of any response
print('Description Length: %d' % max_len)
# prepare sequences
train_descriptions_len = len(train_descriptions)

#do files already exist?
if set(['X1train.npy', 'X2train.npy','ytrain.npy','vocabsize_maxlen.json']) < set(os.listdir('data')):
	X1train, X2train, ytrain = np.load('data/X1train.npy'), np.load('data/X2train.npy'), np.load('data/ytrain.npy')
	with open('data/vocabsize_maxlen.json') as f:
		data = json.load(f)
	vocab_size =  data['vocab_size']
	max_len = data['max_len']
	train_descriptions_len = data['train_descriptions_len']
else:
	"""
	how outputs will be structured
	X1,		X2 (text sequence), 						y (word)
	photo	startseq, 									little
	photo	startseq, little,							girl
	photo	startseq, little, girl, 					running
	photo	startseq, little, girl, running, 			in
	photo	startseq, little, girl, running, in, 		field
	photo	startseq, little, girl, running, in, field, endseq
	"""
	X1train, X2train, ytrain = create_sequences(tokenizer, max_len, train_descriptions, train_features)
	#save
	np.save('data/X1train.npy',X1train)
	np.save('data/X2train.npy',X2train)
	np.save('data/ytrain.npy',ytrain)
	with open('data/vocabsize_maxlen.json', 'w') as outfile:
		json.dump({'vocab_size' : vocab_size, 'max_len' : max_len, 'train_descriptions_len' : len(train_descriptions)}, outfile)
# dev dataset

if set(['X1test.npy', 'X2test.npy','ytest.npy']) < set(os.listdir('data')):
	X1test, X2test, ytest = np.load('data/X1test.npy'), np.load('data/X2test.npy'), np.load('data/ytest.npy')
else:
	X1test, X2test, ytest = create_sequences(tokenizer, max_len, test_descriptions, test_features)
	np.save('data/X1test.npy',X1test)
	np.save('data/X2test.npy',X2test)
	np.save('data/ytest.npy',ytest)
 
# fit model
# data generator, intended to be used in a call to model.fit_generator()
def data_generator(descriptions, photos, tokenizer, max_length):
	# loop for ever over images
	while 1:
		for key, desc_list in descriptions.items():
			# retrieve the photo feature
			photo = photos[key][0]
			in_img, in_seq, out_word = create_sequences(tokenizer, max_length, desc_list, photo)
			yield [[in_img, in_seq], out_word]
 
# define the model
model = define_model(vocab_size, max_len)
# define checkpoint callback
filepath = 'models/model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
"""for generator
# fit model
epochs = 20
steps = train_descriptions_len
for i in range(epochs):
	# create the data generator
	generator = data_generator(train_descriptions, train_features, tokenizer, max_len)
	# fit for one epoch
	model.fit_generator(generator, epochs=1, steps_per_epoch=steps)
	# save model
	model.save('models/model_' + str(i) + '.h5')
"""
#below is for non generoator
model.fit([X1train, X2train], ytrain, epochs=20, callbacks=[checkpoint], validation_data=([X1test, X2test], ytest))