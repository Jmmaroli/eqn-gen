# Test the analysis framework on synthetic and real examples.
# Usage: python run_example.py #

import math
import sys
import numpy as np
import json

from lib.fitting_functions import fitting_functions
from estimate_equation import estimate_equation

if __name__ == "__main__":
    
    if len(sys.argv) == 1:
        # Manually set example (for running in Spyder console).
        # If running in Spyder console, set visual to True.
        EXAMPLE = 1
        print("Usage: python run_example.py #\n")
    else:
        EXAMPLE = int(sys.argv[1])
    print("Running example number {:d}...\n".format(EXAMPLE))

    model_parameters = {
        "epochs": 20,               # upper epoch limit
        "cuda": True,               # use the GPU
        "dropout": 0.0,             # dropout applied to layers
        "clip": -1,                 # -1 means no clip
        "ksize": 2,                 # kernel size
        "levels": 1,                # number of levels
        "nhid": 64,                 # number of hidden units per layer
        "batch_size_train": 32,     # batch size for training
        "train_loss_full": 0,       # [1] full loss, [0] avg over batches
        "lr": 0.002,                # learning rate
        "lr_grad_period": 20,       # decay period for learning rate
        "lr_grad_rate": 0.5,        # decay multiplier after decay period
        "optimizer": 'Adam',        # optimizer to use
        "history": 8,               # number of time sample inputs to the network
        "test_data": 0.2,           # test data proportion
        "seed": 1111,               # random seed
        "visual": False,             # plot training metrics
        "save_visual": True,        # save training metric plot
    }
    
    analysis_parameters = {
        "functions": fitting_functions(),   # An object containing template product functions
        "sweep_initial": 25,                # Elements in initial sweep set for significance detection
        "sweep_detailed": 1000,             # Elements in detailed sweep set for curve fitting
        "contrib_thresh": 0.05,             # Minimum product function significance for inclusion [0.00-1.00]
        "contrib_thresh_omit": 0.10,        # Threshold for conditional product function inclusion
        "use_f_weight": True,               # Use product function weighting
        "seed": 1111,                       # Analysis rng seed for reproducability
        "verbose": True,                    # Print details of analysis
        "visual": False,                     # Plot available 2D and 3D product function samples
        "save_visual": True,                # Save plots of available 2D and 3D product function samples
        "GA_population": 250,               # Population size for GA tuning
        "GA_generations": 100               # Number of generations in GA tuning
    }
    
    # Random number generator seed for reproducability.
    np.random.seed(1234)
    
    if EXAMPLE == 0:
        # Verbose example.
        # y[k] = -0.5*u[k-1] + 0.5*y[k-2]^2 + 0.5*u[k]y[k-1]
        print("Verbose Example\n")
        input_data = 2*(np.random.rand(25000, 2)-0.5)
        output_data = np.zeros([25000, 1])
        
        model_parameters["epochs"] = 10
        model_parameters["ksize"] = 2
        model_parameters["levels"] = 1
        model_parameters["history"] = 2
        model_parameters["cuda"] = False
        
        for i in range(2, 25000):
            u_k0 = input_data[i, 0]
            u_k1 = input_data[i-1, 0]
            y_k1 = output_data[i-1, 0]
            y_k2 = output_data[i-2, 0]
            input_data[i, 1] = y_k1
            input_data[i-1, 1] = y_k2
            
            output_data[i, 0] = -0.5*u_k1 + 0.5*pow(y_k2,2) + 0.5*u_k0*y_k1
            
    elif EXAMPLE == 1:
        # Verbose example extension.
        # y[k] = -0.5*u[k-1] + 0.5*y[k-2]^2 + 0.5*u[k]y[k-1]
        print("Verbose Example Extension\n")
        input_data = 2*(np.random.rand(25000, 2)-0.5)
        output_data = np.zeros([25000, 1])
        
        model_parameters["epochs"] = 20
        model_parameters["ksize"] = 3
        model_parameters["levels"] = 2
        model_parameters["history"] = 5
        model_parameters["cuda"] = False
        
        for i in range(2, 25000):
            u_k0 = input_data[i, 0]
            u_k1 = input_data[i-1, 0]
            y_k1 = output_data[i-1, 0]
            y_k2 = output_data[i-2, 0]
            input_data[i, 1] = y_k1
            input_data[i-1, 1] = y_k2
            
            output_data[i, 0] = -0.5*u_k1 + 0.5*pow(y_k2,2) + 0.5*u_k0*y_k1
            
    elif EXAMPLE == 2:
        # Process noise resiliance.
        # y[k] = -0.5*u[k-1] + 0.5*y[k-2]^2 + 0.5*u[k]y[k-1] + e
        print("Process Noise\n")
        input_data = 2*(np.random.rand(25000, 2)-0.5)
        output_data = np.zeros([25000, 1])
        error = np.zeros(25000)
        
        model_parameters["epochs"] = 20
        model_parameters["ksize"] = 3
        model_parameters["levels"] = 2
        model_parameters["history"] = 5
        model_parameters["cuda"] = False
        
        for i in range(2, 25000):
            u_k0 = input_data[i, 0]
            u_k1 = input_data[i-1, 0]
            y_k1 = output_data[i-1, 0]
            y_k2 = output_data[i-2, 0]
            input_data[i, 1] = y_k1
            input_data[i-1, 1] = y_k2
            
            # Range of 0.00 - 0.16 tested, system unstable at 0.17
            error[i] = np.random.normal(0.0,0.16)
            
            output_data[i, 0] = -0.5*u_k1 + 0.5*pow(y_k2,2) + 0.5*u_k0*y_k1 + error[i]
            
        e_mae = sum(abs(error))/25000
        e_mse = np.mean(pow(error,2))
        e_rmse = np.sqrt(e_mse)
        
        print("Noise Metrics")
        print("MAE: " + str('%.2e'%e_mae))
        print("MSE: " + str('%.2e'%e_mse))
        print("RMSE: " + str('%.2e'%e_rmse))
        print()
                    
    elif EXAMPLE == 3:
        # Measurement noise resiliance.
        # y[k] = -0.5*u[k-1] + 0.5*y[k-2]^2 + 0.5*u[k]y[k-1]
        # Error added after calculation.
        print("Measurement Noise\n")
        input_data = 2*(np.random.rand(25000, 2)-0.5)
        output_data = np.zeros([25000, 1])
        error = np.zeros(25000)
        
        model_parameters["epochs"] = 20
        model_parameters["ksize"] = 3
        model_parameters["levels"] = 2
        model_parameters["history"] = 5
        model_parameters["cuda"] = False
        
        for i in range(2, 25000):
            u_k0 = input_data[i, 0]
            u_k1 = input_data[i-1, 0]
            y_k1 = output_data[i-1, 0]
            y_k2 = output_data[i-2, 0]
            input_data[i, 1] = y_k1
            input_data[i-1, 1] = y_k2
            error[i] = np.random.normal(0.0,2.2)
            
            output_data[i, 0] = -0.5*u_k1 + 0.5*pow(y_k2,2) + 0.5*u_k0*y_k1

        # Error is added after system calculation.
        output_data[:,0] = output_data[:,0] + error
        
        e_mae = sum(abs(error))/25000
        e_mse = np.mean(pow(error,2))
        e_rmse = np.sqrt(e_mse)
        
        o_mae = sum(abs(output_data[:,0]))/25000
        o_mse = np.mean(pow(output_data[:,0],2))
        o_rmse = np.sqrt(o_mse)
        
        print("Noise Metrics")
        print("MAE: " + str('%.2e'%e_mae))
        print("MSE: " + str('%.2e'%e_mse))
        print("RMSE: " + str('%.2e'%e_rmse))
        print()
        print("Output Metrics")
        print("MAE: " + str('%.2e'%o_mae))
        print("MSE: " + str('%.2e'%o_mse))
        print("RMSE: " + str('%.2e'%o_rmse))
        print()
        print("Noise Percentage: " + str('%.2f'%(e_mae/o_mae*100)))
        print()
                
    elif EXAMPLE == 4:
        # Periodic functions.
        # y[k] = 0.5*sin(4*pi*u[k]) + u[k-1] from -1 to 1
        print("Periodic Functions\n")
        input_data = 2*(np.random.rand(25000, 1)-0.5)
        output_data = np.zeros([25000, 1])
        
        model_parameters["epochs"] = 30
        model_parameters["ksize"] = 2
        model_parameters["levels"] = 1
        model_parameters["history"] = 3
        model_parameters["cuda"] = False
        
        analysis_parameters["sweep_initial"] = 1
        
        for i in range(2, 25000):
            u_k0 = input_data[i, 0]
            u_k1 = input_data[i-1, 0]
            
            output_data[i, 0] = 0.5*math.sin(4*math.pi*u_k0) + u_k1
            
    elif EXAMPLE == 5:
        # Modified product functions.
        # Taylor series expansion.
        # y[k] = e^(u[k]) from -1 to 1
        print("Modified Product Functions and Taylor Series Expansion\n")
        input_data = 2*(np.random.rand(25000, 1)-0.5)
        output_data = np.zeros([25000, 1])
        
        model_parameters["epochs"] = 10
        model_parameters["ksize"] = 2
        model_parameters["levels"] = 1
        model_parameters["history"] = 3
        model_parameters["cuda"] = False
            
        for i in range(2, 25000):
            u_k0 = input_data[i, 0]
            u_k1 = input_data[i-1, 0]
            
            output_data[i, 0] = math.exp(u_k0)
            
    elif EXAMPLE == 6:
        # Missing template functions.
        # y[k] = 1/(u[k]+1) from -0.75 to 0.75
        print("Missing Template Functions\n")
        input_data = 0.75*2*(np.random.rand(25000, 1)-0.5)
        output_data = np.zeros([25000, 1])
        
        model_parameters["epochs"] = 10
        model_parameters["ksize"] = 2
        model_parameters["levels"] = 1
        model_parameters["history"] = 3
        model_parameters["cuda"] = False
            
        for i in range(2, 25000):
            u_k0 = input_data[i, 0]
            u_k1 = input_data[i-1, 0]
            
            output_data[i, 0] = 1/(u_k0+1)
    
    elif EXAMPLE == 7:
        # Models undefined at 0.
        # y[k] = u[k]^2 from 1 to 2
        print("Models Undefined at 0\n")
        input_data = np.random.rand(25000, 1)+1
        output_data = np.zeros([25000, 1])
        
        model_parameters["epochs"] = 10
        model_parameters["ksize"] = 2
        model_parameters["levels"] = 1
        model_parameters["history"] = 3
        model_parameters["cuda"] = False
            
        for i in range(2, 25000):
            u_k0 = input_data[i, 0]
            u_k1 = input_data[i-1, 0]
            
            output_data[i, 0] = pow(u_k0,2)
        
    elif EXAMPLE == 8:
        # Silver box example.
        model_parameters["epochs"] = 1000
        model_parameters["ksize"] = 3
        model_parameters["levels"] = 2
        model_parameters["history"] = 5
        model_parameters["nhid"] = 22
        model_parameters["batch_size_train"] = 512
        model_parameters["cuda"] = True
        model_parameters["test_data"] = (127500-85000)/(127500-40585)
        
        analysis_parameters["sweep_initial"] = 250
        analysis_parameters["contrib_thresh"] = 0.02
        analysis_parameters["contrib_thresh_omit"] = 0.05
        analysis_parameters["use_f_weight"] = True
        
        vinArray = []
        voutArray = []
        with open('data/silverbox/SNLS80mV.csv') as fp:
            for cnt, line in enumerate(fp):
                if cnt > 0:
                    splitLine = line.split(',')
                    vin = float(splitLine[0])
                    vout = float(splitLine[1].replace('\n',''))
                    
                    vinArray.append(vin)
                    voutArray.append(vout)
                
            # Use samples 40586-85000 for training and 85001-127500 for validation.
            vinArray = vinArray[40585:127500]
            voutArray = voutArray[40585:127500]
                
            input_data = np.zeros([len(vinArray),2])
            output_data = np.zeros([len(vinArray),1])
            for i in range(1, len(vinArray)):
                input_data[i,0] = vinArray[i-1]
                input_data[i,1] = voutArray[i-1]
                output_data[i,0] = voutArray[i]
                
            input_data = input_data[1:]
            output_data = output_data[1:]
            
    elif EXAMPLE == 9:
        # Roll dynamics example.
        model_parameters["epochs"] = 150
        model_parameters["ksize"] = 2
        model_parameters["levels"] = 2
        model_parameters["history"] = 3
        model_parameters["nhid"] = 32
        model_parameters["cuda"] = False
        model_parameters["test_data"] = 0.10
        model_parameters["dropout"] = 0.10
        
        analysis_parameters["sweep_initial"] = 250
        analysis_parameters["contrib_thresh"] = 0.005
        analysis_parameters["contrib_thresh_omit"] = 0.01
        analysis_parameters["use_f_weight"] = True
        
        roll_array = []
        velocity_array = []
        angle_array = []
        with open('data/roll/roll_data.txt') as fp:
            for cnt, line in enumerate(fp):
                if cnt > 0:
                    split_line = line.split('\t')
                    roll = float(split_line[0])
                    velocity = float(split_line[1])
                    angle = float(split_line[2])
                    
                    roll_array.append(roll)
                    velocity_array.append(velocity)
                    angle_array.append(angle)
                
            input_data = np.zeros([len(roll_array), 3])
            output_data = np.zeros([len(roll_array), 1])
            for i in range(1, len(roll_array)):
                input_data[i,0] = velocity_array[i]
                input_data[i,1] = angle_array[i]
                input_data[i,2] = roll_array[i-1]
                output_data[i,0] = roll_array[i]
            input_data = input_data[1:]
            output_data = output_data[1:]
            
    else:
        print("Invalid example selection. Exiting.")
        sys.exit()
        
    # Print model and analysis parameters for reference.
    print("Model Parameters:")
    print(json.dumps(model_parameters, indent=4, sort_keys=True))
    print()
    print("Analysis Parameters:")
    print(json.dumps({i:analysis_parameters[i] for i in analysis_parameters if i!='functions'},
                      indent=4,
                      sort_keys=True) + "\n")

    # Estimate the equation using input-output data.
    estimate_equation(model_parameters, analysis_parameters, input_data, output_data)
