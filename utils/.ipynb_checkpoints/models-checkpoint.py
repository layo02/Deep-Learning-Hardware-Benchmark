from tensorflow.keras import backend
from tensorflow.keras.applications import VGG19, Xception, ResNet152V2
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D, Dropout, Dense, concatenate
from tensorflow.keras import regularizers
from tensorflow.keras import Model, Input
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam


def model_VGG19():
    ##
	# Define the model's architecture:
	#
	convolution_base = VGG19(weights = 'imagenet', include_top = False, input_shape = (32,32,3))

    convolution_base.trainable = False
    ###
    model = Sequential()
    model.add(convolution_base)
    x = model.output
    ###
    x = Conv2D(128, kernel_size=(3,3), activation = 'relu', padding = 'same')(x)
    x = Conv2D(128, kernel_size=(3,3), activation = 'relu', padding = 'same')(x)
    x = Conv2D(128, kernel_size=(3,3), activation = 'relu', padding = 'same')(x)
    x = Conv2D(128, kernel_size=(3,3), activation = 'relu', padding = 'same')(x)
    x = GlobalAveragePooling2D()(x)
    ###
    x = Dense(4096, activation = 'relu', kernel_regularizer = regularizers.l1_l2(l1 = 1e-5, l2 = 1e-4),
          bias_regularizer = regularizers.l2(1e-4),
          activity_regularizer = regularizers.l2(1e-5))(x)
    x = Dropout(0.5)(x)
    x = Dense(2048, activation = 'relu', kernel_regularizer = regularizers.l1_l2(l1 = 1e-5, l2 = 1e-4),
          bias_regularizer = regularizers.l2(1e-4),
          activity_regularizer = regularizers.l2(1e-5))(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation = 'relu', kernel_regularizer = regularizers.l1_l2(l1 = 1e-5, l2 = 1e-4),
          bias_regularizer = regularizers.l2(1e-4),
          activity_regularizer = regularizers.l2(1e-5))(x)
    out = Dense(10, activation = 'softmax')(x)
    finalModel = Model(inputs = model.input, outputs = out)

    ##
    # Compile the model with defined optimizer and metrics:
    #
    opt = Adam(learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-07)
    finalModel.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['accuracy', 'Precision', 'Recall'])
    
    ##
    # Save the best model based on validation accuracy:
    #
    full_name = 'Model A'
    filepath = "classifiers/%s-{epoch:02d}-{val_accuracy:.4f}-MNIST.hdf5"%full_name
    checkpoint = ModelCheckpoint(filepath, monitor = 'val_accuracy', save_best_only = True, mode = 'max')
    callbacks_list = [checkpoint]
    return finalModel, callbacks_list

def model_concatenate():
    ##
	# Define the model's architecture:	
	#
	
	input_tensor = Input(shape = (32,32,3))
	baseModel1 = Xception(weights = 'imagenet', include_top = False, input_tensor = input_tensor)
	features1 = baseModel1.output
	baseModel2 = ResNet152V2(weights = 'imagenet', include_top = False, input_tensor = input_tensor)
	features2 = baseModel2.output
	concatenated = concatenate([features1, features2])
	###
	x = Conv2D(filters = 1024, kernel_size = (3,3), 
           strides = (1,1), padding = 'same', activation = 'relu')(concatenated)
	###
	x = GlobalAveragePooling2D()(x)
	x = Dense(4096, activation = 'relu', kernel_regularizer = regularizers.l1_l2(l1 = 1e-5, l2 = 1e-4), 
	bias_regularizer = regularizers.l2(1e-4), activity_regularizer = regularizers.l2(1e-5))(x)
	x = Dropout(0.5)(x)
	x = Dense(2048, activation = 'relu', kernel_regularizer = regularizers.l1_l2(l1 = 1e-5, l2 = 1e-4), 
          bias_regularizer = regularizers.l2(1e-4), activity_regularizer = regularizers.l2(1e-5))(x)
		  x = Dropout(0.5)(x)
		  x = Dense(1024, activation = 'relu', kernel_regularizer = regularizers.l1_l2(l1 = 1e-5, l2 = 1e-4), 
          bias_regularizer = regularizers.l2(1e-4), activity_regularizer = regularizers.l2(1e-5))(x)
		  out = Dense(10, activation = 'softmax')(x)
		  finalModel = Model(inputs = input_tensor, outputs = out)
	###
	for layer in concatenated.layers:
		layer.trainable = False
	
	##
	# Compile the model with defined optimizer and metrics:
	#
	opt = Adam(learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-07)
	finalModel.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['accuracy', 'Precision', 'Recall'])
	
	##
	#
	#
	full_name = 'Model B'
    filepath = "classifiers/%s-{epoch:02d}-{val_accuracy:.4f}-MNIST.hdf5"%full_name
    checkpoint = ModelCheckpoint(filepath, monitor = 'val_accuracy', save_best_only = True, mode = 'max')
    callbacks_list = [checkpoint]
    
	return finalModel, callbacks_list