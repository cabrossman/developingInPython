#https://blog.floydhub.com/checkpointing-tutorial-for-tensorflow-keras-and-pytorch/

#callbacks -- to save model performance

"""
You can employ different checkpoint strategies according to the type of experiment training 
regime you're performing:

Short Training Regime (minutes to hours)
---save checkpoint at end of training or epoch

Normal Training Regime (hours to day) 
---save checkpoints every n_epochs OR keep track of best one
---restricting maximum checkpoints to 10, where the new ones replace the earliest ones

Long Training Regime (days to weeks)
---similar to Normal
"""

"""
FREQUENCY	CHECKPOINTS		CONS												PRO
High		High			You need a lot of space!!							You can resume very quickly in almost all the interesting training states
High		Low				You could have lost precious states					Minimize the storage space you need
Low			High			It will take time to get to intermediate states		You can resume the experiments in a lot of interesting states
Low			Low				You could have lost precious states					Minimize the storage space you need
"""
import os
dirr = 'C:\\Users\\chrisb\\OneDrive - Leesa\\jobs\\developingInPython\\keras_callbacks'
os.chdir(dirr)
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Flatten, Dense, Dropout, Dense
from tensorflow.keras.callbacks import Callback, TensorBoard, CSVLogger, ReduceLROnPlateau, ModelCheckpoint, LambdaCallback

import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model =  Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

#callbacks
checkpoint = ModelCheckpoint('./saved_models/bestmodel.hdf5',monitor='val_acc',verbose=1,save_best_only=True,mode='max',period = 2)
csvlogger = CSVLogger('./saved_models/test.csv', separator=',', append=False)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=5, min_lr=0.001) #reduce learning rate
tb = TensorBoard(log_dir='./tmp/logs', histogram_freq=1, batch_size=32, write_graph=True)
import json
json_log = open('./saved_models/loss_log.json', mode='wt', buffering=1)
json_logging_callback = LambdaCallback(
    on_epoch_end=lambda epoch, logs: json_log.write(
        json.dumps({'epoch': epoch, 'loss': logs['loss']}) + '\n'),
    on_train_end=lambda logs: json_log.close()
)

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

history = LossHistory()

callback_list = [checkpoint,csvlogger,reduce_lr, tb, json_logging_callback, history]


model_history = model.fit(x_train, y_train, epochs=10, callbacks=callback_list, validation_split=0.02)
model.evaluate(x_test, y_test)

#plot the history -- same history goes to CSV file
plt.plot(model_history.history['acc'])
plt.plot(model_history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

#load best saved model
model = load_model('./saved_models/bestmodel.hdf5')