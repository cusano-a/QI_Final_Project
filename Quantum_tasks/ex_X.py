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


# Monitores performances per batch (training set only)
class LossHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.loss = []
        self.acc = []

    def on_batch_end(self, batch, logs={}):
        self.loss.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))


save_dir = os.path.join(os.getcwd(), 'saved_results_ising')
model_name = 'ising_trained_model.h5'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)

seed=123
train_frac=0.7
val_frac=0.2

# Acquires the data
data_1=np.load('evec_dis_2000.npy')[:,1:]	# 0 < |lambda| < 1
data_2=np.load('evec_tra_2000.npy')[:,1:]	# 1 < |lambda| < 2	
data_3=np.load('evec_ord_2000.npy')[:,1:]	# 2 < |lambda| < 3
data=np.concatenate((data_1,data_2,data_3), axis=0)

labels_1=np.ones(data_1.shape[0], dtype=int)
labels_2=np.zeros(data_2.shape[0], dtype=int)
labels_3=np.zeros(data_3.shape[0], dtype=int)
labels=np.concatenate((labels_1,labels_2,labels_3), axis=0)	

Ntrain=int(train_frac*data.shape[0])
Nval=int(val_frac*data.shape[0])
Ntest=data.shape[0]-Ntrain-Nval

print('Data acquired! \n')

# Random shuffles the data
np.random.seed(seed)
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices] # Shuffle an array by row
labels = labels[indices]
						
# Splits the data in training, validation, test sets
data=data.reshape(data.shape[0],data.shape[1],1)
train_data=data[:Ntrain,:,:]
val_data=data[Ntrain:(Ntrain+Nval),:,:]
test_data=data[-Ntest:,:,:]
train_lab=labels[:Ntrain]
val_lab=labels[Ntrain:(Ntrain+Nval)]
test_lab=labels[-Ntest:]

print("Total entries: {}\n Training:{}\n Validation: {}\n Test: {}". \
	format(data.shape[0], Ntrain, Nval, Ntest))

# Building the model
print('Building the model ...')
model = Sequential()

model.add(Flatten(input_shape=(1024,1)))
model.add(Dense(64))
model.add(Activation('relu'))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc'])

print('Done! \n')

inp=raw_input('Print a summary of the model? (y/n)\t')
if(inp=='y'):
	model.summary()

# Trains the model
raw_input("Press any key to start the training \n")

batch_history=LossHistory() 
history = model.fit(train_data,
                    train_lab,
                    epochs=10,
                    batch_size=128,
                    validation_data=(val_data, val_lab),
                    callbacks=[batch_history],
                    verbose=1)  

# Saves the model and the results
model.save(model_path)

history_dict = history.history
history_dict.keys()  
acc = np.array(history_dict['acc'])
val_acc = np.array(history_dict['val_acc'])
loss = np.array(history_dict['loss'])
val_loss = np.array(history_dict['val_loss'])

time_ep = range(1, len(acc) + 1)
time_ba = range(1, len(batch_history.acc) + 1)

history_path=os.path.join(save_dir, 'history_ising.txt')
historyb_path=os.path.join(save_dir, 'historyb_ising.txt')
np.savetxt(history_path, np.transpose((time_ep, acc, val_acc, 
												loss, val_loss)))
np.savetxt(historyb_path, np.transpose((time_ba, 
										batch_history.acc, 
										batch_history.loss)))
											
# Prints the results  		                       
print('Final validation accuracy: {}, loss: {}'.format(
		val_acc[-1], val_loss[-1]))  
test_loss, test_acc = model.evaluate(test_data, test_lab)
print('Test accuracy:', test_acc)               

# Graphs of accuracy and loss over time (in epochs and batches)
plt.figure(1)
plt.xlim(left=1, right=10)
plt.plot(time_ep, loss, 'r', linewidth=1.5, label='Training')
plt.plot(time_ep, val_loss, 'b', linewidth=1.5, label='Validation')
plt.title('Loss on Ising', fontsize=16)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.legend()
plt.savefig('loss_ising.pdf')

plt.figure(2)
plt.xlim(left=1, right=10)
plt.plot(time_ep, acc*100.0, 'r', linewidth=1.5, label='Training')
plt.plot(time_ep, val_acc*100.0, 'b', linewidth=1.5, label='Validation')
plt.title('Accuracy on Ising', fontsize=16)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Accuracy (%)', fontsize=14)
plt.legend()
plt.savefig('acc_ising.pdf')

plt.figure(3)
plt.plot(time_ba, batch_history.loss, 'r', 
			linewidth=1.5, label='Training')
plt.title('Loss per batch on Ising', fontsize=16)
plt.xlabel('Batch', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.legend()
plt.savefig('batch_loss_ising.pdf')

plt.figure(4)
plt.plot(time_ba, np.array(batch_history.acc)*100.0, 'r', 
			linewidth=1.5, label='Training')
plt.title('Accuracy per batch on Ising', fontsize=16)
plt.xlabel('Batch', fontsize=14)
plt.ylabel('Accuracy (%)', fontsize=14)
plt.legend()
plt.savefig('batch_acc_ising.pdf')
