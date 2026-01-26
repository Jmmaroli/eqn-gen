# Generate a TCN model from input output data.
#
# Input and output are time series, each row is a time entry and each column
# is a dimension of the data. Most recent data is the last column.
#
# Example input:                Example output:
# x1[k-n] x2[k-n] x3[k-n]       y1[k-n] y2[k-n]
# ...     ...     ...           ...     ...
# x1[k-2] x2[k-2] x3[k-2]       y1[k-2] y2[k-2]
# x1[k-1] x2[k-1] x3[k-1]       y1[k-1] y2[k-1]
# x1[k]   x2[k]   x3[k]         y1[k]   y2[k]
#
# Training data is rearanged into input 'X_train' and output 'Y_train'.
# Each row of 'X_train' is a sample of length 'history' and dimension
# 'input_channels' in the following format (ex: history = 4):
#
# x1[k-3] x1[k-2] x1[k-1] x1[k]
# x2[k-3] x2[k-2] x2[k-1] x2[k]
# x3[k-3] x3[k-2] x3[k-1] x3[k]
#
# Note that this matrix is flipped right-left in the dissertation.
#
# Each row of 'Y_train' is a sample of length 1 and dimension
# 'output_channels' in the following format:
# y1[k]
# y2[k]
#
# The goal of the TCN is to predict y[k] given x[k] .. x[k-n]
#
# The mask is in the same format as 'X_train' and determines which inputs
# are hidden from the TCN. Element-wise multiplication is used to set input
# elements to 0 with the mask.

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from lib.model import TCN
import pyprind # Progress bar

FORMAT = '%.3e'

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def create_model(model_parameters, inputData, outputData, inputMask=1):
    
    epochs = model_parameters["epochs"]                       # upper epoch limit
    cuda = model_parameters["cuda"]                           # use the GPU
    dropout = model_parameters["dropout"]                     # dropout applied to layers
    clip = model_parameters["clip"]                           # -1 means no clip
    ksize = model_parameters["ksize"]                         # kernel size
    levels = model_parameters["levels"]                       # number of levels
    nhid = model_parameters["nhid"]                           # number of hidden units per layer
    batch_size_train = model_parameters["batch_size_train"]   # batch size for training
    train_loss_full = model_parameters["train_loss_full"]     # [1] full loss, [0] avg over batches
    lr = model_parameters["lr"]                               # learning rate
    lr_grad_period = model_parameters["lr_grad_period"]       # decay period for learning rate
    lr_grad_rate = model_parameters["lr_grad_rate"]           # decay multiplier after decay period
    optimizer = model_parameters["optimizer"]                 # optimizer to use
    history = model_parameters["history"]                     # number of time sample inputs to the network
    test_data = model_parameters["test_data"]                 # test data proportion
    seed = model_parameters["seed"]                           # random seed
    visual = model_parameters["visual"]                       # plot training metrics
    save_visual = model_parameters["save_visual"]             # save training metric plot

    torch.manual_seed(seed)
        
    # Get the current data output folder if saving data and plots.
    if save_visual:
        if not os.path.exists('./output'):
            os.mkdir('./output')
        model_dir_count = 1
        while os.path.exists('./output/model_{}'.format(model_dir_count)):
            model_dir_count = model_dir_count + 1
        os.mkdir('./output/model_{}'.format(model_dir_count))

    # Create the TCN network.
    print("Initializing TCN model...")
    history_eff = (ksize-1)*(pow(2,levels+1)-1)-(ksize-1) + 1
    input_length = inputData.shape[0]
    output_length = outputData.shape[0]
    if input_length != output_length:
        print("Input and output data must be the same length.")
        exit
    input_channels = inputData.shape[1]     # dimension of x[k]
    output_channels = outputData.shape[1]   # dimension of y[k]
    channel_sizes = [nhid]*levels
    model = TCN(input_channels, output_channels, channel_sizes, ksize, dropout=dropout)
    print("The network has ", levels, " hidden levels with a kernel size of ", ksize, sep='')
    print("The oldest input that can be seen is x[k-", history_eff-1, "]", sep='')
    if history_eff > history: print("WARNING: Effective history exceeds input history,")
    if history_eff > history: print("         inputs beyond x[k-" + str(history-1) + "] are not visible")
    print("The network contains " + str(count_parameters(model)) + " parameters")
    print()
    
    # Format the training and test data.
    samples = input_length - history + 1
    train_samples = int(np.round((1-test_data)*samples))
    test_samples = samples-train_samples
    X_train = np.zeros([train_samples, input_channels, history])
    Y_train = np.zeros([train_samples, output_channels])
    X_test = np.zeros([test_samples, input_channels, history])
    Y_test = np.zeros([test_samples, output_channels])
    input_range = []
    input_shift = []
    shiftedInputData = np.copy(inputData)
    for i in range(0,input_channels):
        minRange = min(inputData[:,i])
        maxRange = max(inputData[:,i])
        if minRange <= 0 and maxRange >= 0:
            # No shifting required, the zero point is in the input range.
            input_range.append([minRange, maxRange])
            input_shift.append(0)
        else:
            # Adjust the input channel to center it about zero.
            shift = (minRange + maxRange)/2
            shiftedInputData[:,i] = shiftedInputData[:,i] - shift
            input_shift.append(shift)
            input_range.append([minRange-shift, maxRange-shift])
    for i in range(0,train_samples):
        X_train[i,:,:] = shiftedInputData[i:(i+history)].T
        Y_train[i,:] = outputData[i+history-1].T    
    for i in range(train_samples,samples):
        X_test[i-train_samples,:,:] = shiftedInputData[i:(i+history)].T
        Y_test[i-train_samples,:] = outputData[i+history-1].T
        
    # Normalize data to be mean centered with unit covariance.
    # Apply input mask.
    mu_x = np.mean(X_train,axis=0)
    sig_x = np.var(X_train,axis=0)
    mu_y = np.mean(Y_train,axis=0)
    sig_y = np.var(Y_train,axis=0)
    mu_y_t = torch.tensor(mu_y, dtype=torch.float)
    sig_y_t = torch.tensor(sig_y, dtype=torch.float)
    X_train = torch.tensor(((X_train-mu_x)/sig_x)*inputMask,dtype=torch.float)
    Y_train = torch.tensor((Y_train-mu_y)/sig_y,dtype=torch.float)
    X_test = torch.tensor(((X_test-mu_x)/sig_x)*inputMask,dtype=torch.float)
    Y_test = torch.tensor((Y_test-mu_y)/sig_y,dtype=torch.float)

    # Move data to the GPU if one is present.
    if cuda:
        print("CUDA specified in model parameters.")
        if torch.cuda.is_available():
            model.cuda()
            X_train = X_train.cuda()
            Y_train = Y_train.cuda()
            X_test = X_test.cuda()
            Y_test = Y_test.cuda()
            mu_y_t = mu_y_t.cuda()
            sig_y_t = sig_y_t.cuda()
            print("Model moved to GPU.")
        else:
            print("WARNING: GPU unavailable, model will run on the CPU.")
            cuda = False
    else:
        print("CUDA not specified, model will run on the CPU.")

    # Define the training function.
    optimizer = getattr(optim, optimizer)(model.parameters(), lr=lr)
    def train(epoch):
        model.train()
        batch_idx = 1
        epoch_loss = 0
        for i in range(0, X_train.size()[0], batch_size_train):
            if i + batch_size_train > X_train.size()[0]:
                x, y = X_train[i:], Y_train[i:]
            else:
                x, y = X_train[i:(i+batch_size_train)], Y_train[i:(i+batch_size_train)]
            optimizer.zero_grad()
            output = model(x)
            loss = F.mse_loss(output*sig_y_t+mu_y_t, y*sig_y_t+mu_y_t)
            loss.backward()
            if clip > 0:
                torch.nn.utils.clip_grad_norm(model.parameters(), clip)
            optimizer.step()
            batch_idx += 1
            epoch_loss += loss.data.item()
        return epoch_loss / (batch_idx - 1)

    # Define test loss function.
    def evaluateTestLoss():
        model.eval()
        output = model(X_test)
        test_loss = F.mse_loss(output*sig_y_t+mu_y_t, Y_test*sig_y_t+mu_y_t)
        return test_loss.data.item()

    # Train the network
    print("Training TCN model...")
    progress_bar = pyprind.ProgBar(2*epochs, monitor=True)
    testLossHistory = list()
    trainLossHistory = list()
    for ep in range(1, epochs+1):
        trainloss = train(ep)
        testloss = evaluateTestLoss()
        if train_loss_full == 1:
            # Calculate training loss over all training data. More accurate but uses more RAM.
            if cuda : model.cpu()
            trainloss = F.mse_loss(model(X_train.cpu())*sig_y_t+mu_y_t, Y_train.cpu()*sig_y_t+mu_y_t)
            if cuda : model.cuda()
        testLossHistory.append(testloss)
        trainLossHistory.append(trainloss)
        if ep % lr_grad_period == 0:
            lr = lr*lr_grad_rate
        progress_bar.update()
        progress_bar.update()
    time.sleep(0.5)
    print()
            
    # Plot training and test loss.
    if save_visual == True or visual == True:
        plt.figure()
        ax = plt.subplot(1,1,1)
        plt.plot(trainLossHistory)
        plt.plot(testLossHistory)
        plt.legend(['Training Loss','Test Loss'])
        plt.title('Training and Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        ax.set_yscale("log", nonpositive='clip')
        if save_visual == True: plt.savefig('./output/model_{}/loss.pdf'.format(model_dir_count))
        if visual == True: plt.show()
    print("Min train: epoch " + str(np.argmin(trainLossHistory)+1))
    print("Min test: epoch " + str(np.argmin(testLossHistory)+1))
    print()
    
    # Calculate data fit metrics on the CPU.
    model.cpu()
    X_data = torch.cat([X_train.cpu(), X_test.cpu()], dim=0)
    Y_data = torch.cat([Y_train.cpu(), Y_test.cpu()], dim=0)
    Y_pred = np.zeros([samples, output_channels])
    Y_pred = torch.tensor(Y_pred, dtype=torch.float)
    batch_size_test = 128
    for i in range(0, samples, batch_size_test):
        if i + batch_size_test > samples:
            Y_pred[i:] = model(X_data[i:])
        else:
            Y_pred[i:(i+batch_size_test)] = model(X_data[i:(i+batch_size_test)])
    Y_err = (Y_data.detach().cpu().numpy()*sig_y+mu_y) - (Y_pred.detach().cpu().numpy()*sig_y+mu_y)
    Y_err_train = Y_err[0:train_samples]
    Y_err_test = Y_err[train_samples:]
    mae = []
    mse = []
    rmse = []
    for i in range(0, output_channels):
        mae.append({})
        mse.append({})
        rmse.append({})
        mae[i]["total"] = (np.mean(abs(Y_err[:, i])))
        mae[i]["train"] = (np.mean(abs(Y_err_train[:, i])))
        mae[i]["test"] = (np.mean(abs(Y_err_test[:, i])))
        mse[i]["total"] = (sum(pow(Y_err[:, i], 2))/len(Y_err[:, i]))
        mse[i]["train"] = (sum(pow(Y_err_train[:, i], 2))/len(Y_err_train[:, i]))
        mse[i]["test"] = (sum(pow(Y_err_test[:, i], 2))/len(Y_err_test[:, i]))
        rmse[i]["total"] = np.sqrt(mse[i]["total"])
        rmse[i]["train"] = np.sqrt(mse[i]["train"])
        rmse[i]["test"] = np.sqrt(mse[i]["test"])
        print("TCN channel " + str(i+1) + " metrics")
        print("Total MAE: " + str(FORMAT%mae[i]["total"]) + " MSE: " + \
              str(FORMAT%mse[i]["total"]) + " RMSE: " + str(FORMAT%rmse[i]["total"]))
        print("Train MAE: " + str(FORMAT%mae[i]["train"]) + " MSE: " + \
              str(FORMAT%mse[i]["train"]) + " RMSE: " + str(FORMAT%rmse[i]["train"]))
        print("Test  MAE: " + str(FORMAT%mae[i]["test"]) + " MSE: " + \
              str(FORMAT%mse[i]["test"]) + " RMSE: " + str(FORMAT%rmse[i]["test"]))
    print()
    
    # Assemble the function output.
    modelDictionary = {
        "model": model,
        "model_parameters": model_parameters,
        "history_eff": history_eff,
        "mu_x": mu_x,
        "sig_x": sig_x,
        "mu_y": mu_y,
        "sig_y": sig_y,
        "input_channels": input_channels,
        "output_channels": output_channels,
        "input_range": input_range,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "input_shift": input_shift
    }
            
    return modelDictionary
