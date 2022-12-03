import numpy as np

name = ["12022022_222700","12022022_230042"]

label = []
data = []

for na in name:
    label.extend(np.load("label "+na+".npy", allow_pickle=True))
    data.extend(np.load("data "+na+".npy", allow_pickle=True))
loadeddata = np.array(data,dtype=np.uint8)
loadedlabel = np.array(label,dtype=np.uint8)


from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten,Dropout,Convolution2D,MaxPooling2D
from tensorflow.python.keras.losses import mean_squared_error; 
from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD


model =  Sequential([
        Dense(1, activation='relu', input_shape=(1,20,10,1)),
        Flatten(),
        Dense(25, activation='relu'),
        Dense(25, activation='relu'),
        Dense(25, activation='relu'),
        Dense(5, activation='relu'),
    ])

print(model.summary())

model.compile(optimizer=SGD(learning_rate=1e-2),loss= mean_squared_error)
model.build()

model.fit(loadeddata,loadedlabel,epochs=10,verbose=1)     


print("trained")