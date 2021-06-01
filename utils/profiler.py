import tensorflow
import numpy as np

def memory_usage(model, batch_size, *args, **kwargs):
    """
    Return the estimated memory usage of a given Keras model in gbytes.
    This includes the model weights and layers, but excludes the dataset.

    The model shapes are multipled by the batch size, but the weights are not.

    Args:
        model: A Keras model.
        batch_size: The batch size you intend to run the model with. If you
            have already specified the batch size in the model itself, then
            pass `1` as the argument here.
    Returns:
        An estimate of the Keras model's memory usage in MB.

    """
    default_dtype = tensorflow.keras.backend.floatx()
    
    ##
    # Estimate the memory according to the number of layers inside the model:
    #
    shapes_mem_count = 0
    internal_model_mem_count = 0

    for layer in model.layers:
        if isinstance(layer, tensorflow.keras.Model):
            internal_model_mem_count += memory_usage(layer, batch_size=batch_size)
            
        single_layer_mem = tensorflow.as_dtype(layer.dtype or default_dtype).size
        out_shape = layer.output_shape
        
        if isinstance(out_shape, list):
            out_shape = out_shape[0]
        for s in out_shape:
            if s is None:
                continue
            single_layer_mem *= s
            
        shapes_mem_count += single_layer_mem
    
    ##
    # Estimate the memory usage according to the number of weights:
    #
    trainable_count = sum([tensorflow.keras.backend.count_params(p) for p in model.trainable_weights])
    non_trainable_count = sum([tensorflow.keras.backend.count_params(p) for p in model.non_trainable_weights])
    
    ##
    # TOTAL MEMORY = MEMORY_ALL_LAYERS + MEMORY_ALL_WEIGHTS + INTERNAL_MODEL_MEMORY
    #
    if tensorflow.keras.backend.floatx() == 'float16':
        number_size = 2.0
        
    if tensorflow.keras.backend.floatx() == 'float32':
        number_size = 4.0
        
    if tensorflow.keras.backend.floatx() == 'float64':
        number_size = 8.0

    total_memory = number_size * (batch_size * shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3) + internal_model_mem_count
    return gbytes

def memory_weights(model, *args, **kwargs):
    """
    Return the memory required to store model weights in mbytes.
    
    Args:
        model: a Keras model.
    Returns:
        total_memory: the required memory.
        
    """
    ##
    # Count the number of parameters:
    #
    total_memory = model.count_params()
    
    ##
    # Compute the exact required memory based on policy:
    #
    if tensorflow.keras.backend.floatx() == 'float16':
        total_memory = total_memory * 2.0
    
    if tensorflow.keras.backend.floatx() == 'float32':
        total_memory = total_memory * 4.0
    
    if tensorflow.keras.backend.floatx() == 'float64':
        total_memory = total_memory * 8.0
        
    total_memory = np.round(total_memory / (1024.0 ** 2),3)
    return total_memory
