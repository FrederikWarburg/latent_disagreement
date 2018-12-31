#from models.nalu import NALU
import torch.nn.functional as F
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm_notebook as tqdm

def generate_synthetic_selection_dataset(min_value, max_value, sample_size, set_size, boundaries = None):
    """
    generates a dataset of integers for the synthetics arithmetic task
    :param arithmetic_op: the type of operation to perform on the sum of the two sub sections can be either :
    ["add" , "subtract", "multiply", "divide", "root", "square"]
    :param min_value: the minimum possible value of the generated integers
    :param max_value: the maximum possible value of the generated integers
    :param sample_size: the number of integers per sample
    :param set_size: the number of samples in the dataset
    :param boundaries: [Optional] an iterable of 4 integer indices in the following format :
    [start of 1st section, end of 1st section, start of 2nd section, end of 2nd section]
    if None, the boundaries are randomly generated.
    :return: the training dataset input, the training true outputs, the boundaries of the sub sections used
    """
    x = np.random.uniform(min_value, max_value, (set_size, sample_size))
    

    
    if boundaries is None:
        boundaries = [np.random.randint(sample_size) for i in range(4)]
        boundaries = sorted(boundaries)
        
        if boundaries[1] == boundaries[0]:
            if boundaries[1] < sample_size-1:
                boundaries[1] =boundaries[1]+ 1
            else:
                boundaries[0] = boundaries[0] - 1
                
        if boundaries[3] == boundaries[2]:
            if boundaries[3] < sample_size-1:
                boundaries[3] =boundaries[3]+ 1
            else:
                boundaries[2] =boundaries[2]- 1
    else:
        if len(boundaries) != 4:
            raise ValueError("boundaries is expected to be a list of 4 elements but found {}".format(len(boundaries)))

    a = np.array([np.sum(sample[boundaries[0]:boundaries[1]]) for sample in x])
    b = np.array([np.sum(sample[boundaries[2]:boundaries[3]]) for sample in x])

    return torch.Tensor(x), torch.Tensor(np.array([a,b]).T), boundaries

def generate_synthetic_operation_dataset(arithmetic_op, min_value, max_value, sample_size, set_size, boundaries = None):
    """
    generates a dataset of integers for the synthetics arithmetic task

    :param arithmetic_op: the type of operation to perform on the sum of the two sub sections can be either :
    ["add" , "subtract", "multiply", "divide", "root", "square"]
    :param min_value: the minimum possible value of the generated integers
    :param max_value: the maximum possible value of the generated integers
    :param sample_size: the number of integers per sample
    :param set_size: the number of samples in the dataset
    :param boundaries: [Optional] an iterable of 4 integer indices in the following format :
    [start of 1st section, end of 1st section, start of 2nd section, end of 2nd section]
    if None, the boundaries are randomly generated.
    :return: the training dataset input, the training true outputs, the boundaries of the sub sections used
    """
    #scaled_input_values = np.random.uniform(min_value, max_value, (set_size, sample_size))
    '''
    if boundaries is None:
        boundaries = [np.random.randint(sample_size) for i in range(4)]
        boundaries = sorted(boundaries)
        
        if boundaries[1] == boundaries[0]:
            if boundaries[1] < sample_size-1:
                boundaries[1] =boundaries[1]+ 1
            else:
                boundaries[0] = boundaries[0] - 1
                
        if boundaries[3] == boundaries[2]:
            if boundaries[3] < sample_size-1:
                boundaries[3] =boundaries[3]+ 1
            else:
                boundaries[2] =boundaries[2]- 1
    else:
        if len(boundaries) != 4:
            raise ValueError("boundaries is expected to be a list of 4 elements but found {}".format(len(boundaries)))
    
    a = np.array([np.sum(sample[boundaries[0]:boundaries[1]]) for sample in scaled_input_values])
    b = np.array([np.sum(sample[boundaries[2]:boundaries[3]]) for sample in scaled_input_values])
    '''
    a = np.random.uniform(min_value,max_value,(set_size,1))
    b = np.random.uniform(min_value,max_value,(set_size,1))
    
    true_outputs = None
    if "add" in str.lower(arithmetic_op):
        true_outputs = a + b
    elif "sub" in str.lower(arithmetic_op):
        true_outputs = a - b
    elif "mult" in str.lower(arithmetic_op):
        true_outputs = a * b
    elif "div" in str.lower(arithmetic_op):
        true_outputs = a / b
    elif "square" == str.lower(arithmetic_op):
        true_outputs = a * a
    elif "root" in str.lower(arithmetic_op):
        true_outputs = np.sqrt(a)
    scaled_input_values = torch.tensor(np.concatenate((a,b),axis=1),dtype=torch.float32)
    #scaled_input_values = torch.tensor(scaled_input_values, dtype=torch.float32)
    true_outputs = torch.tensor(true_outputs, dtype=torch.float32)
        
    return scaled_input_values, true_outputs