import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

def rnn():
    ##
    # Create a model:
    #
    model = Sequential()
    model.add(LSTM(2816, input_shape = (32,32), activation = 'relu', time_major = True, return_sequences = True))
    model.add(LSTM(2560, activation = 'relu'))
    model.add(LSTM(1760, activation = 'relu'))
    model.add(LSTM(1024, activation ='relu'))
    model.add(Dense(10, activation = 'softmax'))
    
    ##
    # Compile the model:
    #
    opt = Adam(learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-07)
    model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['accuracy', 'Precision', 'Recall'])
    return model
