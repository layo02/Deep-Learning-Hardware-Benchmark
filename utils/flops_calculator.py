import tensorflow
from tensorflow.keras import backend


def get_flops(model):
    session = tensorflow.compat.v1.Session()
    graph = tensorflow.compat.v1.get_default_graph()

    with graph.as_default():
        with session.as_default():
            model = finalModel
            
            run_meta = tensorflow.compat.v1.RunMetadata()
            opts = tensorflow.compat.v1.profiler.ProfileOptionBuilder.float_operation()

            ##
            # Use the Keras session graph in the call to the profiler:
            #
            flops = tensorflow.compat.v1.profiler.profile(graph = graph,
                                                  run_meta = run_meta, cmd='op', options=opts)

    tensorflow.compat.v1.reset_default_graph()

    return flops.total_float_ops