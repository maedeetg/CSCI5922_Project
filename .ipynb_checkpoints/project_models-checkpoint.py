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
from sklearn.preprocessing import OneHotEncoder
import tensorflow.keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import seaborn as sns

#######################################################
def one_hot_encode(y, num_labels):
    labels = ["L", "R", "U", "D", "B", "F"] 
    
    # Create a mapping dictionary
    labels_to_num = {"L":1, "R":2, "U":3, "D":4, "B":5, "F":6}
    
    # Convert list using the dictionary
    numeric_list = [labels_to_num[tuple(char)[0]] for char in y]
    
    n = len(numeric_list)
    one_hot_labels = np.zeros((n, num_labels))

    for i in range(n):
        one_hot_labels[i, int(numeric_list[i]) - 1] = 1

    return(one_hot_labels)

#######################################################
# Read in data
train = pd.read_csv("training.csv")
test = pd.read_csv("testing.csv")
valid = pd.read_csv("validation.csv")
labels = ["L", "R", "U", "D", "B", "F"] 

# Filter the DataFrame to include only specific labels
train = train[train['Action'].isin(labels)]
test = test[test['Action'].isin(labels)]
valid = valid[valid['Action'].isin(labels)]

# Split into labels and data
x_train = train.loc[:, train.columns != "Action"].to_numpy() 
x_test = test.loc[:, train.columns != "Action"].to_numpy()
x_valid = test.loc[:, valid.columns != "Action"].to_numpy()
y_train = train.loc[:, test.columns == "Action"].to_numpy()
y_test = test.loc[:, test.columns == "Action"].to_numpy()
y_valid = test.loc[:, valid.columns == "Action"].to_numpy()

# What does the data look like?
print("The shape of x_train is: ", x_train.shape)
print("The shape of y_train is: ", y_train.shape)
print("The shape of x_test is: ", x_test.shape)
print("The shape of y_test is: ", y_test.shape)
print("The shape of x_valid is: ", x_valid.shape)
print("The shape of y_valid is: ", y_valid.shape)

print("The first value of x_train is: \n", x_train[0])
print("The first value of y_train is: ", y_train[0])
print("The first value of y_train is: ", y_train[0])
print("The first value of x_test is: \n", x_test[0])
print("The first value of y_test is: ", y_test[0])
print("The first value of x_valid is: ", x_valid[0])
print("The first value of y_valid is: ", y_valid[0])

# One-hot-encode labels
num_labels = len(np.unique(labels))
y_train_one_hot = one_hot_encode(y_train, num_labels)
y_test_one_hot = one_hot_encode(y_test, num_labels)
y_valid_one_hot = one_hot_encode(y_valid, num_labels)

# Need to change datatypes so they are compataible for keras later
x_train = np.array(x_train, dtype=np.float32) 
y_train = np.array(y_train_one_hot, dtype=np.float32) 
x_test = np.array(x_test, dtype=np.float32)
y_test = np.array(y_test_one_hot, dtype=np.float32)
x_valid = np.array(x_valid, dtype=np.float32)
y_valid = np.array(y_valid_one_hot, dtype=np.float32)

#######################################################
# # Create sequential ANN. 
# # BUILD MODEL
# ANN = keras.Sequential([tf.keras.layers.Flatten(input_shape=(54, )),
#                         tf.keras.layers.Dense(100, activation='relu'),
#                         tf.keras.layers.Dropout(0.5),
#                         tf.keras.layers.Dense(50, activation='relu'),
#                         tf.keras.layers.Dropout(0.2),
#                         tf.keras.layers.Dense(25, activation='relu'),
#                         tf.keras.layers.Dropout(0.2),
#                         tf.keras.layers.Dense(6, activation='softmax')])

# # MODEL SUMMARY
# ANN.summary()

# # COMPILE MODEL
# ANN.compile(loss="categorical_crossentropy",
#             metrics=["accuracy"],
#             optimizer='adam')

# # FIT THE MODEL TO TRAINING DATA
# Fit = ANN.fit(x_train, y_train, epochs = 100, validation_data = (x_valid, y_valid))

# # PLOT RESULTS
# # Accuracy
# plt.plot(Fit.history['accuracy'], label = 'training accuracy', color = 'magenta')
# plt.plot(Fit.history['val_accuracy'], label = 'validation accuracy', color = 'purple')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.title("Accuracy over Epochs")
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
# labels = ["L", "R", "U", "D", "B", "F"] 
# cm = confusion_matrix(y_numeric, y_hat_numeric)
# ax = plt.subplot()
# sns.heatmap(cm, annot = True, fmt = 'g', ax = ax, cmap = 'flare')  
# ax.set_xlabel("Predicted labels")
# ax.set_ylabel("True labels")
# ax.set_title("Confusion Matrix")
# ax.xaxis.set_ticklabels(labels)
# ax.yaxis.set_ticklabels(labels)
# plt.show()
#######################################################
# Create sequential CNN. 
# Need to reshape the x_train and x_test to be used for a CNN. 
# I will reshape as (6, 3, 3) instead of the (54). I picked this since there are 6 sides to a rubiks
# cube and each side is 3 by 3
x_train_reshape = x_train.reshape(3542, 6, 3, 3)
x_test_reshape = x_test.reshape(393, 6, 3, 3)
x_valid_reshape = x_valid.reshape(393, 6, 3, 3)

# BUILD MODEL
CNN = keras.Sequential([
    tf.keras.layers.Conv2D(input_shape = (6, 3, 3), kernel_size = (2, 2), filters = 128, activation = 'relu', strides = (1, 1), padding = "same"), 
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(kernel_size = (2, 2), filters = 64, activation = 'relu', strides = (1, 1), padding = "same"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(6, activation = "softmax")])

CNN.summary()

# COMPILE MODEL
CNN.compile(loss="categorical_crossentropy",
            metrics=["accuracy"],
            optimizer='adam')


# FIT THE MODEL TO TRAINING DATA
Fit = CNN.fit(x_train_reshape, y_train, epochs = 100, validation_data = (x_valid_reshape, y_valid))

# PLOT RESULTS
# Accuracy
plt.plot(Fit.history['accuracy'], label = 'training accuracy', color = 'magenta')
plt.plot(Fit.history['val_accuracy'], label = 'validation accuracy', color = 'purple')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title("Accuracy over Epochs")
plt.legend(loc='lower right')
plt.show()
# Loss
plt.plot(Fit.history['loss'], label = 'training loss', color = 'magenta')
plt.plot(Fit.history['val_loss'], label = 'validation loss', color = 'purple')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title("Loss over Epochs")
plt.legend(loc='lower right')
plt.show()

# TEST
Test_Loss, Test_Accuracy = CNN.evaluate(x_test_reshape, y_test)

# PREDICT & CONFUSION MATRIX
predictions = CNN.predict([x_test_reshape])
Max_Values = np.squeeze(np.array(predictions.argmax(axis=1))) # all our label predictions

y_numeric = np.argmax(y_test, axis = 1)     
y_hat_numeric = Max_Values
labels = ["L", "R", "U", "D", "B", "F"] 
cm = confusion_matrix(y_numeric, y_hat_numeric)
ax = plt.subplot()
sns.heatmap(cm, annot = True, fmt = 'g', ax = ax, cmap = 'flare')  
ax.set_xlabel("Predicted labels")
ax.set_ylabel("True labels")
ax.set_title("Confusion Matrix")
ax.xaxis.set_ticklabels(labels)
ax.yaxis.set_ticklabels(labels)
plt.show()
#######################################################
# Create sequential CNN. 
# Need to reshape the x_train and x_test to be used for a CNN. 
# I will reshape as (6, 3, 3) instead of the (54). I picked this since there are 6 sides to a rubiks
# cube and each side is 3 by 3
# x_train_reshape = x_train.reshape(2039, 6, 3, 3)
# red = x_train_reshape[:][:,0]
# yellow = x_train_reshape[:][:,1]
# green = x_train_reshape[:][:,2]
# white = x_train_reshape[:][:,3]
# orange = x_train_reshape[:][:,4]
# blue = x_train_reshape[:][:,5]
# space = -1*np.ones((2039, 3, 3)) 

# x_train_reshape2 = np.array([[space, yellow, space], [blue, red, green], [space, white, space], [space, orange, space]])
# print(red)
# print(yellow)
# x_test_reshape = x_test.reshape(1022, 6, 3, 3)

# # BUILD MODEL
# CNN = keras.Sequential([
#     tf.keras.layers.Conv2D(input_shape = (6, 3, 3), kernel_size = (2, 2), filters = 128, activation = 'relu', strides = (1, 1), padding = "same"), 
#     tf.keras.layers.Dropout(0.5),
#     tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
#     tf.keras.layers.Conv2D(kernel_size = (2, 2), filters = 64, activation = 'relu', strides = (1, 1), padding = "same"),
#     tf.keras.layers.Dropout(0.5),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(6, activation = "softmax")])

# CNN.summary()

# # COMPILE MODEL
# CNN.compile(loss="categorical_crossentropy",
#             metrics=["accuracy"],
#             optimizer='adam')


# # FIT THE MODEL TO TRAINING DATA
# Fit = CNN.fit(x_train_reshape, y_train, epochs = 50, validation_data = (x_test_reshape, y_test))

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
# Test_Loss, Test_Accuracy = CNN.evaluate(x_test_reshape, y_test)

# # PREDICT & CONFUSION MATRIX
# predictions = CNN.predict([x_test])
# Max_Values = np.squeeze(np.array(predictions.argmax(axis=1))) # all our label predictions

# y_numeric = np.argmax(y_test, axis = 1)     
# y_hat_numeric = Max_Values
# labels = ["L", "R", "U", "D", "B", "F"] 
# cm = confusion_matrix(y_numeric, y_hat_numeric)
# ax = plt.subplot()
# sns.heatmap(cm, annot = True, fmt = 'g', ax = ax, cmap = 'flare')  
# ax.set_xlabel("Predicted labels")
# ax.set_ylabel("True labels")
# ax.set_title("Confusion Matrix")
# ax.xaxis.set_ticklabels(labels)
# ax.yaxis.set_ticklabels(labels)
# plt.show()