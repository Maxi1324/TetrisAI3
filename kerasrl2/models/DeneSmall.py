from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten

def model(windowsize):
    return Sequential([
        Dense(1, activation='relu', input_shape=(windowsize,20,10,1)),
        Flatten(),
        Dense(100, activation='relu'),
        Dense(200, activation='relu'),
        Dense(100, activation='relu'),
        Dense(5, activation='relu'),
    ])