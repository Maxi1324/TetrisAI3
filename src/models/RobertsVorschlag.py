from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten

def model(windowsize):
    return  Sequential([
        Dense(2, activation='relu', input_shape=(windowsize,20,10,2)),
        Flatten(),

        Dense(1000, activation='relu'),
        Dense(1000, activation='relu'),
        Dense(1000, activation='relu'),
        Dense(1000, activation='relu'),

        Dense(3, activation='relu')
    ])