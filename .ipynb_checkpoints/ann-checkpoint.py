# Import libraries
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import tensorflow.keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import seaborn as sns

#######################################################
# ANN with all data
# Read in data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Split into labels and data
x_train = train.loc[:, train.columns != "vocals"].to_numpy() 
x_train = x_train[:, 4:]
x_test = test.loc[:, train.columns != "vocals"].to_numpy()
x_test = x_test[:, 4:]
y_train = train.loc[:, test.columns == "vocals"].to_numpy()
y_test = test.loc[:, test.columns == "vocals"].to_numpy()

# What does the data look like?
print("The first value of x_train is: \n", x_train[0])
print("The shape of x_train is: ", x_train.shape)

print("The first value of y_train is: ", y_train[0])
print("The shape of y_train is: ", y_train.shape)

print("The first value of x_test is: \n", x_test[0])
print("The shape of x_test is: ", x_test.shape)

print("The first value of y_test is: ", y_test[0])
print("The shape of y_test is: ", y_test.shape)

# Create sequential ANN. 
# BUILD MODEL
ANN = keras.Sequential([tf.keras.layers.Flatten(input_shape=(7, )),
                                tf.keras.layers.Dropout(0.5),
                                tf.keras.layers.Dense(100, activation='relu'),
                                tf.keras.layers.Dense(2, activation='sigmoid')])

# # MODEL SUMMARY
# ANN.summary()

# # COMPILE MODEL
# ANN.compile(loss="sparse_categorical_crossentropy",
#             metrics=["accuracy"],
#             optimizer='adam')

# # FIT THE MODEL TO TRAINING DATA
# Fit = ANN.fit(x_train, y_train, epochs = 50, validation_data = (x_test, y_test))

# # PLOT RESULTS
# # Accuracy
# plt.plot(Fit.history['accuracy'], label = 'training accuracy', color = 'magenta')
# plt.plot(Fit.history['val_accuracy'], label = 'validation accuracy', color = 'purple')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.title("Accuracy over Epochs")
# plt.ylim([0.5, 1])
# plt.legend(loc='lower right')
# plt.show()
# # Loss
# plt.plot(Fit.history['loss'], label = 'training loss', color = 'magenta')
# plt.plot(Fit.history['val_loss'], label = 'validation loss', color = 'purple')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title("Loss over Epochs")
# plt.legend(loc='lower right')
# plt.show()

# # TEST
# Test_Loss, Test_Accuracy = ANN.evaluate(x_test, y_test)

# # PREDICT & CONFUSION MATRIX
# predictions = ANN.predict([x_test])
# Max_Values = np.squeeze(np.array(predictions.argmax(axis=1))) # all our label predictions

# y_numeric = np.argmax(y_test, axis = 1)     
# y_hat_numeric = Max_Values
# labels = ['football', 'politics', 'science']
# cm = confusion_matrix(y_numeric, y_hat_numeric)
# ax = plt.subplot()
# sns.heatmap(cm, annot = True, fmt = 'g', ax = ax, cmap = 'flare')  
# ax.set_xlabel("Predicted labels")
# ax.set_ylabel("True labels")
# ax.set_title("Confusion Matrix")
# ax.xaxis.set_ticklabels(labels)
# ax.yaxis.set_ticklabels(labels)
# plt.show()
# #######################################################