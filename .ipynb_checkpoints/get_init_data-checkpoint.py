import numpy as np

def get_inputs(f_num):
    inputs = np.load(f'initial_data/function_{f_num}/initial_inputs.npy')
    return np.array(inputs, dtype=object)
    
def get_outputs(f_num):
    return np.load(f'initial_data/function_{f_num}/initial_outputs.npy')