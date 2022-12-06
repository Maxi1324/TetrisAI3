from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten,Convolution2D

def model(windowsize):
    return Sequential([
        Convolution2D(32,(1,1),activation="relu",input_shape=(windowsize,20,10,1)),
        Convolution2D(64,(1,1),activation="relu"),
        Convolution2D(64,(1,1),activation="relu"),
        Flatten(),
        Dense(150, activation='relu'),
        Dense(50, activation='relu'),
        Dense(5, activation='relu'),
    ])