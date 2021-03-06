import tensorflow
from tensorflow.keras.layers import Conv2D, Dense, AveragePooling2D, Flatten
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam

def convolve2D(IMG_SIZE1,IMG_SIZE2,IMG_CH,*args):
    ##
    # Create a model:
    #
    model = Sequential()
    model.add(Conv2D(2048, input_shape = (IMG_SIZE1,IMG_SIZE2,IMG_IMG_CH), kernel_size = (3,3), strides = (2,2), padding = 'same'))
    model.add(AveragePooling2D(pool_size = (2,2), strides = (2,2), padding = 'same'))
    model.add(Conv2D(128, kernel_size = (3,3), strides = (2,2), padding = 'same'))
    model.add(AveragePooling2D(pool_size=(2,2), strides =(2,2), padding = 'same'))
    model.add(Conv2D(64, kernel_size = (3,3), strides = (2,2), padding = 'same'))
    model.add(AveragePooling2D(pool_size=(2,2), strides =(2,2), padding = 'same'))
    model.add(Flatten())
    model.add(Dense(10, activation = 'softmax'))
    
    ##
    # Compile and deploy the model:
    #
    opt = Adam(learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-07)
    model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['accuracy', 'Precision', 'Recall'])
    
    return model
