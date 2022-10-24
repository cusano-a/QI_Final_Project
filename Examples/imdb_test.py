from __future__ import absolute_import, division, print_function
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Activation, Embedding
from tensorflow.python.keras.layers import GlobalAveragePooling1D
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os


# Downloads the IMDB dataset (bug in original file)
load_imdb = keras.datasets.imdb		# (see load_imdb.py)

# Keeps the top 10000 most frequent words in the training data
(train_data, train_labels), (test_data, test_labels)= \
	load_imdb.load_data(num_words=10000)
print('Data acquired! \n')
print("Training entries: {}, labels: {}".\
	format(len(train_data), len(train_labels)))

# A dictionary mapping words to an integer index 
word_index = load_imdb.get_word_index()
word_index = {k:(v+3) for k,v in word_index.items()} 
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  
word_index["<UNUSED>"] = 3
reverse_word_index = \
	dict([(value, key) for (key, value) in word_index.items()])
def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

# Reads a review as example
user_input=input('Which review has to be shown? (write an int<25000)\n')
rev = int(user_input)
print(decode_review(train_data[rev]), train_labels[rev])

# Pads the arrays so they all have the same length
train_data = pad_sequences(train_data,
							value=word_index["<PAD>"],
							padding='post',
							maxlen=256)
test_data = pad_sequences(test_data,
						   value=word_index["<PAD>"],
						   padding='post',
						   maxlen=256)

# Building the model
print('Building the model ...')

vocab_size = 10000
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=16))
model.add(GlobalAveragePooling1D())
model.add(Dense(units=16))
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

# Training the model
raw_input("Press any key to start the training \n")
# Splits the data in training set and validation set 
x_val = train_data[:10000]
partial_x_train = train_data[10000:]
y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=2)                   
history_dict = history.history
history_dict.keys()                    

# Evaluate the predictions on validation set
results = model.evaluate(test_data, test_labels)
print('On the test set the results are: \n')
print("Loss {}, accuracy {}".format(results[0], results[1]))                   

# Graphs of accuracy and loss over epochs
acc = np.array(history_dict['acc'])
val_acc = np.array(history_dict['val_acc'])
loss = np.array(history_dict['loss'])
val_loss = np.array(history_dict['val_loss'])

time = range(1, len(acc) + 1)
np.savetxt('history_imdb.txt', 
			np.transpose((time, acc, val_acc, loss, val_loss)))

plt.figure(1)
plt.xlim(left=1, right=40)
plt.plot(time, loss, 'r', linewidth=1.5, label='Training')
plt.plot(time, val_loss, 'b', linewidth=1.5, label='Validation')
plt.title('Loss on IMDb')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_imdb.pdf')

plt.figure(2)
plt.xlim(left=1, right=40)
plt.plot(time, acc*100.0, 'r', linewidth=1.5,label='Training')
plt.plot(time, val_acc*100.0, 'b', linewidth=1.5, label='Validation')
plt.title('Accuracy on IMDb', fontsize=16)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Accuracy (%)', fontsize=14)
plt.legend()
plt.savefig('acc_imdb.pdf')
  
