from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LSTM
from numpy import array

"""Return Sequence"""

# define model
inputs1 = Input(shape=(3, 1))
"""
	setting return_sequences=True will return a hidden value for each time step
	This is required when stacking LSTMs
	You may also need to access the sequence of hidden state outputs when predicting
		a sequence of outputs with a Dense output layer wrapped in a TimeDistributed layer.
"""
lstm1 = LSTM(1, return_sequences=True)(inputs1)
model = Model(inputs=inputs1, outputs=lstm1)
# define input data
data = array([0.1, 0.2, 0.3]).reshape((1,3,1))
# make and show prediction -- hidden state
print(model.predict(data))


"""
	Return State

Keras provides the return_state argument to the LSTM layer that will provide 
access to the hidden state output (state_h) and the cell state (state_c). 
"""
inputs1 = Input(shape=(3, 1))
lstm1, state_h, state_c = LSTM(1, return_state=True)(inputs1)
model = Model(inputs=inputs1, outputs=[lstm1, state_h, state_c])
# define input data
data = array([0.1, 0.2, 0.3]).reshape((1,3,1))
# make and show prediction
print(model.predict(data))

"""return both state and sequence"""
# define model
inputs1 = Input(shape=(3, 1))
lstm1, state_h, state_c = LSTM(1, return_sequences=True, return_state=True)(inputs1)
model = Model(inputs=inputs1, outputs=[lstm1, state_h, state_c])
# define input data
data = array([0.1, 0.2, 0.3]).reshape((1,3,1))
# make and show prediction
print(model.predict(data))