from __future__ import absolute_import, division, print_function
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Activation
from tensorflow.python.keras.layers import Conv1D, MaxPooling1D, Flatten
import numpy as np
import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


save_dir = os.path.join(os.getcwd(), 'saved_results_wave')
model_name = 'separable_trained_model.h5'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)

seed=123
train_frac=0.7
val_frac=0.2

# Acquires the data
data_1=np.load('sep.npy')
data_2=np.load('nsep.npy')
data=np.concatenate((data_1,data_2), axis=0)

labels_1=np.zeros(data_1.shape[0], dtype=int)
labels_2=np.ones(data_2.shape[0], dtype=int)
labels=np.concatenate((labels_1,labels_2), axis=0)	

print('Data acquired! \n')

# Random shuffle the data
np.random.seed(seed)
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices] # Shuffle an array by row
labels = labels[indices]
						
# Split the data in training, validation, test
Ntrain=int(train_frac*data.shape[0])
Nval=int(val_frac*data.shape[0])
Ntest=data.shape[0]-Ntrain-Nval

train_data=data[:Ntrain,:,:]
val_data=data[Ntrain:(Ntrain+Nval),:,:]
test_data=data[-Ntest:,:,:]
train_lab=labels[:Ntrain]
val_lab=labels[Ntrain:(Ntrain+Nval)]
test_lab=labels[-Ntest:]

print("Total entries: {}\n Training:{}\n Validation: {}\n Test: {}". \
	format(data.shape[0], Ntrain, Nval, Ntest))
print(len(test_lab))

# Building the model
print('Building the model ...')
model = Sequential()
"""
# Model 1, discarded
model.add(Flatten(input_shape=(1024,2)))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(128))
model.add(Activation('relu'))
"""
# Model 2
model.add(Conv1D(32, 3, padding='same', input_shape=(1024,2)))
model.add(Activation('relu'))
model.add(Conv1D(32, 3))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(1))
model.add(Activation('sigmoid'))

opt = tf.keras.optimizers.Adam(lr=0.0015, decay=6e-6)
model.compile(optimizer=opt,
              loss='binary_crossentropy',
              metrics=['acc'])

print('Done! \n')

inp=raw_input('Print a summary of the model? (y/n)\t')
if(inp=='y'):
	model.summary()

# Trains the model
raw_input("Press any key to start the training \n")

history = model.fit(train_data,
                    train_lab,
                    epochs=10,
                    batch_size=128,
                    validation_data=(val_data, val_lab),
                    verbose=1)  

# Saves the model
history_dict = history.history
history_dict.keys()  
acc = np.array(history_dict['acc'])
val_acc = np.array(history_dict['val_acc'])
loss = np.array(history_dict['loss'])
val_loss = np.array(history_dict['val_loss'])
time_ep = range(1, len(acc) + 1)
history_path=os.path.join(save_dir, 'history_wave.txt')
np.savetxt(history_path, np.transpose((time_ep, acc, val_acc, 
												loss, val_loss)))
												
# Prints the results  		                       
print('Final validation accuracy: {}, loss: {}'.format(
		val_acc[-1], val_loss[-1]))  
test_loss, test_acc = model.evaluate(test_data, test_lab)
print('Test accuracy:', test_acc)               

# Graphs of accuracy and loss over time
plt.figure(1)
plt.xlim(left=1, right=10)
plt.plot(time_ep, loss, 'r', linewidth=1.5, label='Training')
plt.plot(time_ep, val_loss, 'b', linewidth=1.5, label='Validation')
plt.title('Loss on separability', fontsize=16)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.legend()
plt.savefig('loss_wave.pdf')

plt.figure(2)
plt.xlim(left=1, right=10)
plt.plot(time_ep, acc*100.0, 'r', linewidth=1.5, label='Training')
plt.plot(time_ep, val_acc*100.0, 'b', linewidth=1.5, label='Validation')
plt.title('Accuracy on separability', fontsize=16)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Accuracy (%)', fontsize=14)
plt.legend()
plt.savefig('acc_wave.pdf')


