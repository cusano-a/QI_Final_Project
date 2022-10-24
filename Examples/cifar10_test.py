from __future__ import absolute_import, division, print_function
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Activation
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten
import os


num_classes = 10
save_dir = os.path.join(os.getcwd(), 'saved_results')
model_name = 'cifar10_trained_model.h5'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

# Loads the CIFAR-10 images dataset
load_cifar10 = keras.datasets.cifar10 # (see load_cifar10.py)
(train_imag, train_lab), (test_imag, test_lab)=load_cifar10.load_data()

print('Data acquired! \n')
print("Training images: {}, labels: {}".\
	format(train_imag.shape[0], train_lab.shape[0]))

# Convert class vectors to one-hot encoding 
train_lab = keras.utils.to_categorical(train_lab, num_classes)
test_lab = keras.utils.to_categorical(test_lab, num_classes)

# Rescale the RGB values in the range [-0.5:0.5]
train_imag = train_imag.astype('float32')
test_imag = test_imag.astype('float32')
train_imag = train_imag/255.0-0.5
test_imag = test_imag /255.0-0.5

# Model of CNN
model = Sequential()
model.add(Conv2D(filters=32,	
				 kernel_size=(3, 3),	
				 padding='same',	
				 input_shape=train_imag.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(units=512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
              
inp=raw_input('Print a summary of the model? (y/n)\t')
if(inp=='y'):
	model.summary()

# Training of the model
raw_input("Press any key to start the training \n")
history = model.fit(
		train_imag,
		train_lab,
		epochs=30,
		batch_size=32,
		validation_data=(test_imag, test_lab),
		verbose=1)
				
# Save the model and the weights
inp=raw_input('Save the model? (y/n)\n')
if(inp=='y'):
	model_path = os.path.join(save_dir, model_name)
	model.save(model_path)
	print('Saved trained model at %s ' % model_path)

history_dict = history.history
history_dict.keys()  
acc = np.array(history_dict['acc'])
val_acc = np.array(history_dict['val_acc'])
loss = np.array(history_dict['loss'])
val_loss = np.array(history_dict['val_loss'])

time = range(1, len(acc) + 1)
history_path=os.path.join(save_dir, 'history_cifar10.txt')
np.savetxt(history_path, 
			np.transpose((time, acc, val_acc, loss, val_loss)))
	
# Print and plot the results  		                       
print('Final validation accuracy: {}, loss: {}'. \
		format(val_acc[-1], val_loss[-1]))
		
plt.figure(1)		
plt.xlim(left=1, right=30)
plt.plot(time, loss, 'r', linewidth=1.5, label='Training')
plt.plot(time, val_loss, 'b', linewidth=1.5, label='Validation')
plt.title('Loss on CIFAR-10',  fontsize=16)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.legend()
plt.savefig(os.path.join(save_dir, 'loss_cifar10.pdf'))

plt.figure(2)
plt.xlim(left=1, right=30)
plt.plot(time, acc*100.0, 'r', linewidth=1.5,label='Training')
plt.plot(time, val_acc*100.0, 'b', 	linewidth=1.5, label='Validation')
plt.title('Accuracy on CIFAR-10', fontsize=16)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Accuracy (%)', fontsize=14)
plt.legend()
plt.savefig(os.path.join(save_dir, 'acc_cifar10.pdf'))

		
