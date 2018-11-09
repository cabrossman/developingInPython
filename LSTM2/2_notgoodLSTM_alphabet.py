#https://machinelearningmastery.com/understanding-stateful-lstm-recurrent-neural-networks-python-keras/

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.utils import to_categorical

# define the raw dataset
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
# create mapping of characters to integers (0-25) and the reverse
char_to_int = dict((c, i) for i, c in enumerate(alphabet))
int_to_char = dict((i, c) for i, c in enumerate(alphabet))


# prepare the dataset of input to output pairs encoded as integers
seq_length = 3
dataX = []
dataY = []
for i in range(0, len(alphabet) - seq_length, 1):
	seq_in = alphabet[i:i + seq_length]
	seq_out = alphabet[i + seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])
	print(seq_in, '->', seq_out)

# reshape X to be [samples, time steps, features]
X = np.reshape(dataX, (len(dataX), 1, seq_length)) """WROPNG!"""

# normalize
X = X / float(len(alphabet))
# one hot encode the output variable
y = to_categorical(dataY)


# create and fit the model
model = Sequential()
model.add(LSTM(32, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=500, batch_size=1, verbose=2)

# summarize performance of the model
scores = model.evaluate(X, y, verbose=0)
print("Model Accuracy: %.2f%%" % (scores[1]*100))


# demonstrate some model predictions
for pattern in dataX:
	x = np.reshape(pattern, (1, 1, len(pattern)))
	x = x / float(len(alphabet)) #normalize
	prediction = model.predict(x, verbose=0) #predict
	index = np.argmax(prediction) #get top prediction
	result = int_to_char[index] #convert index to character
	seq_in = int_to_char[pattern[0]] #get original
	print(seq_in, "->", result) #print
