import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

def rnn():
    model = Sequential()
    model.add(LSTM(2816, input_shape = (x_train.shape[1:]), activation = 'relu', time_major = True, return_sequences = True))
    model.add(LSTM(2560, activation = 'relu'))
    model.add(LSTM(1760, activation = 'relu'))
    model.add(LSTM(1024, activation ='relu'))
    model.add(Dense(10, activation = 'softmax'))
    return model
