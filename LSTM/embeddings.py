#http://adventuresinmachinelearning.com/word2vec-keras-tutorial/
"""
word embeddings try to “compress” large one-hot word vectors into much 
smaller vectors (a few hundred elements) which preserve some of the 
meaning and context of the word. Word2Vec is the most common process 
of word embedding

The context of the word is the key measure of meaning that is utilized in 
Word2Vec.  The context of the word “sat” in the sentence “the cat sat on
 the mat” is (“the”, “cat”, “on”, “the”, “mat”).  In other words, it is 
 the words which commonly occur around the target word “sat”. Words which
 have similar contexts share meaning under Word2Vec, and their reduced 
 vector representations will be similar.  In the skip-gram model version 
 of Word2Vec (more on this later), the goal is to take a target word 
 i.e. “sat” and predict the surrounding context words.  
 This involves an iterative learning process.
 
 he end product of this learning will be an embedding layer in a network 
 – this embedding layer is a kind of lookup table – 
 the rows are vector representations of each word in our vocabulary.
 
 The idea of the neural network above is to supply our input target 
 words as one-hot vectors.  Then, via a hidden layer, we want to train 
 the neural network to increase the probability of valid context words, 
 while decreasing the probability of invalid context words 
 (i.e. words that never show up in the surrounding context of the
 target words).  This involves using a softmax function on the output 
 layer.  Once training is complete, the output layer is discarded, and
 our embedding vectors are the weights of the hidden layer.
 
 skip-gram - uses target to predict sourrounding context words
 CBOX - takes sourounding words to predict target
 
"""

#https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/
"""
For example, below we define an Embedding layer with 
--a vocabulary of 200 (e.g. integer encoded words from 0 to 199, inclusive)
--a vector space of 32 dimensions in which words will be embedded
--an input documents that have 50 words each.



e = Embedding(200, 32, input_length=50)
"""
import numpy as np
from tensorflow.keras.preprocessing.text import one_hot, Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense

# define documents
docs = ['Well done!',
		'Good work',
		'Great effort',
		'nice work',
		'Excellent!',
		'Weak',
		'Poor effort!',
		'not good',
		'poor work',
		'Could have done better.']
# define class labels
labels = np.array([1,1,1,1,1,0,0,0,0,0])

vocab_size = 50
encoded_docs = [one_hot(d, vocab_size) for d in docs]
print(encoded_docs) ## need to pad to all have same length

# pad documents to a max length of 4 words
max_length = 4
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
print(padded_docs)

#The Embedding has a vocabulary of 50 and an input length of 4. We will choose a small embedding space of 8 dimensions.
model = Sequential()
model.add(Embedding(input_dim = vocab_size, output_dim=8, input_length=max_length))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# summarize the model
print(model.summary())


# fit the model
model.fit(padded_docs, labels, epochs=500, verbose=1)
# evaluate the model
loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
print('Accuracy: %f' % (accuracy*100))

