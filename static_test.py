#from models.nalu import NALU
import torch.nn.functional as F
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm_notebook as tqdm
from models.nalu import NALU
import torch
import torchvision.utils as vutils
import numpy as np
import torchvision.models as models
from torchvision import datasets
from tensorboardX import SummaryWriter
import datetime
import os


FEATURES_NUM = 100
epochs = 500
batch_size = 1
lr = 0.01



import cv2
def make_Im(W):

    height,width = np.shape(W)
    new_height = height* 10
    
    im = np.zeros((3,new_height,width))
    w_tmp = W.detach().numpy()
    k = -1

    for i in range(new_height):
        if i % (new_height // 2) == 0:
            k += 1
        
        for channel in range(3):
            im[channel,i,:] = w_tmp[k,:]

    im = np.round(im * 255,0).astype(int)
    return im





def generate_synthetic_arithmetic_dataset(arithmetic_op, min_value, max_value, sample_size, set_size, boundaries = None):
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
    scaled_input_values = np.random.uniform(min_value, max_value, (set_size, sample_size))

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
    
    scaled_input_values = torch.tensor(scaled_input_values, dtype=torch.float32)
    true_outputs = torch.tensor(true_outputs, dtype=torch.float32).unsqueeze(1)
        
    return scaled_input_values, true_outputs, boundaries





interpolations = [True, False]
for interpolation in interpolations:
    
    if interpolation:
        training_range = [0,10]
        test_range = [0,10]
    else:
        training_range = [0,10]
        test_range = [0,100]
    
    operators = ['add', 'sub', 'mult','div','square','root']
    for operator in operators:
        
        first_losses = []
        last_losses = []
        
        for i in range(1):
            stop_training = False
            np.random.seed(i)

            in_dim = FEATURES_NUM
            hidden_dim = 2
            out_dim = 1
            num_layers = 2

            dim = in_dim # dimensition for generating data

            model = NALU(num_layers, in_dim, hidden_dim, out_dim)

            X_train, y_train, boundaries = generate_synthetic_arithmetic_dataset(operator, training_range[0], training_range[1], FEATURES_NUM, 1000)
            X_test, y_test, _ = generate_synthetic_arithmetic_dataset(operator, test_range[0], test_range[1], FEATURES_NUM, 1000, boundaries)
            print(boundaries)
            optimizer = torch.optim.RMSprop(model.parameters(),lr=lr)

            experiment = "int_"+str(interpolation)+",op_" + operator + ",lr_" + str(lr) + ",bs_" + str(batch_size) + ",f_" + str(FEATURES_NUM) +",tr_r_" + str(training_range) + ",te_r_" +str(test_range)
            exp = "exp=" + str(i) + "/"
            Nalues = [exp + 'SelectorNALU/', exp + 'operatorNalu/']

            path = 'checkpoints/' + experiment + '/'
            if not os.path.exists('checkpoints'):
                os.mkdir('checkpoints')
            if not os.path.exists('checkpoints/' + experiment):
                os.mkdir('checkpoints/' + experiment)

            writer = SummaryWriter(path)

            writer.add_text(exp + "text/seed", str(i), 0)
            writer.add_text(exp + "text/boundaries", str(boundaries), 0)

            model.eval()
            output_test = model(X_test)
            loss = F.mse_loss(output_test, y_test)
            first_losses.append(loss.item())

            for epoch in tqdm(range(epochs)):

                for batch in range(len(X_train) // batch_size):

                    model.train()
                    optimizer.zero_grad()

                    X_batch_train = X_train[batch:(batch+batch_size),:]
                    y_batch_train = y_train[batch:(batch+batch_size),:]

                    out = model(X_batch_train)

                    loss = F.mse_loss(out, y_batch_train)

                    loss.backward()
                    optimizer.step()

                if epoch % 10 == 1:

                    writer.add_scalar(exp + "data/Training_loss", loss.item(), epoch)

                    for i, child in enumerate(model.model.children()):

                        g = child.g_store

                        if i == 0:
                            writer.add_graph(Nalues[i]+"hist/g",g,epoch)

                        for j, param in enumerate(child.parameters()):

                            if j == 0:
                                G = param
                            if j == 1:
                                W_hat = param
                            if j == 2:
                                M_hat = param

                        W = torch.tanh(W_hat) * torch.sigmoid(M_hat)
                                                                        
                        writer.add_histogram(Nalues[i]+"hist/W",W,epoch)
                        writer.add_histogram(Nalues[i]+"hist/G",G,epoch)
                        writer.add_histogram(Nalues[i]+"hist/W_hat",W_hat,epoch)
                        writer.add_histogram(Nalues[i]+"hist/M_hat",M_hat,epoch)

                        if i == 0:

                            tmp_im = make_Im(W)
                            writer.add_image(Nalues[i] + "image/W", tmp_im, epoch)

                            tmp_im = make_Im(G)
                            writer.add_image(Nalues[i] + "image/G", tmp_im, epoch)

                            tmp_im = make_Im(W_hat)
                            writer.add_image(Nalues[i] + "image/W_hat", tmp_im, epoch)

                            tmp_im = make_Im(M_hat)
                            writer.add_image(Nalues[i] + "image/M_hat", tmp_im, epoch)

                    model.eval()

                    X_batch_test = X_test[batch:(batch+batch_size),:]
                    y_batch_test = y_test[batch:(batch+batch_size),:]

                    output_test = model(X_batch_test)
                    loss = F.mse_loss(output_test, y_batch_test)

                    acc = np.sum(np.isclose(output_test.detach().numpy(), y_batch_test.detach().numpy(), atol=.1, rtol=0)) / len(y_batch_test)

                    writer.add_scalar(exp + "data/Test_accuracy", float(acc), epoch)
                    writer.add_scalar(exp + "data/Test_loss", loss.item(), epoch)
                                        
            model.eval()

            output_test = model(X_batch_test)
            loss = F.mse_loss(output_test, y_batch_test)
            last_losses.append(loss.item())
                        
            writer.add_text(exp + "data/first_losses", str(first_losses), 0)
            writer.add_text(exp + "data/first_losses_mean", str(np.mean(first_losses)), 0)
            writer.add_text(exp + "data/last_losses", str(last_losses), 0)
            writer.add_text(exp + "data/last_losses_mean", str(np.mean(last_losses)), 0)
    