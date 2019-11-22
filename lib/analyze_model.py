# Analyze a model to generate an equation.
#
# Input is the model, template fitting functions, and the sweep set.
# The sweep set is an array of multiples to use in the fitting process.

import time
import itertools
import multiprocessing
import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import curve_fit
import pyprind # Progress bar
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mat4py
import os

from lib.evaluate_function import evaluate_function

# Format of function parameters.
FORMAT = '%.3e'

def analyze_model(analysis_parameters, model_dictionary, input_data, output_data, input_mask=1):
    
    functions = analysis_parameters["functions"]
    sweep_initial = analysis_parameters["sweep_initial"]
    sweep_detailed = analysis_parameters["sweep_detailed"]
    contrib_thresh = analysis_parameters["contrib_thresh"]
    contrib_thresh_omit = analysis_parameters["contrib_thresh_omit"]
    use_f_weight = analysis_parameters["use_f_weight"]
    seed = analysis_parameters["seed"]
    np.random.seed(seed)
    verbose = analysis_parameters["verbose"]
    visual = analysis_parameters["visual"]
    save_visual = analysis_parameters["save_visual"]
    
    # Check inputs for validity.
    if sweep_initial < 1:
        print("ERROR: analyze_model parameter sweep_initial must be >= 1")
        return None, None
    if sweep_detailed < 100:
        print("ERROR: analyze_model parameter sweep_detailed must be >= 100")
        return None, None
    
    # Function for indexing the large impulse array.
    def array_to_int(num_list):         # [1,2,3]
        str_list = map(str, num_list)   # ['1','2','3']
        num_str = ''.join(str_list)     # '123'
        num = int(num_str, 2)           # 123
        return num
    
    model = model_dictionary["model"]
    history = model_dictionary["model_parameters"]["history"]
    history_eff = model_dictionary["history_eff"]
    mu_x = model_dictionary["mu_x"]
    sig_x = model_dictionary["sig_x"]
    mu_y = model_dictionary["mu_y"]
    sig_y = model_dictionary["sig_y"]
    input_channels = model_dictionary["input_channels"]
    output_channels = model_dictionary["output_channels"]
    input_range = model_dictionary["input_range"]
    input_shift = model_dictionary["input_shift"]

    # Establish tensor types of certain variables for computation.
    mu_y_t = torch.tensor(mu_y, dtype=torch.float)
    sig_y_t = torch.tensor(sig_y, dtype=torch.float)
    
    # Get the current data output folder if saving data and plots.
    if save_visual == True:
        if not os.path.exists('./output'):
            os.mkdir('./output')
        analysis_dir_count = 1
        while os.path.exists('./output/analysis_{}'.format(analysis_dir_count)):
            analysis_dir_count = analysis_dir_count + 1
        os.mkdir('./output/analysis_{}'.format(analysis_dir_count))

    # Generate every possible combination of impulses.
    if history < history_eff:
        history_eff = history
    combination_count = pow(2, input_channels*(history_eff))
    combinations = [x for x in range(0, input_channels*(history_eff))]
    impulse_array = np.zeros([combination_count, input_channels, history])
    # Loop through every combination of subsets of constituants
    for combination_id in range(0, len(combinations)+1):
        for subset in itertools.combinations(combinations, combination_id):
            impulse = np.zeros([1, 1, input_channels*(history_eff)])
            for element in subset:
                impulse[0, 0, input_channels*(history_eff)-1-element] = 1
            index = array_to_int(impulse[0, 0, :].astype(int))
            impulse_shaped = np.reshape(impulse, [input_channels, history_eff])
            # Add buffer elements to account for a history longer than scope.
            impulse_array[index, :, (history-history_eff):history] = impulse_shaped
    
    # Generate the impulse sweep set for creating multiples of impulses.
    if sweep_initial != 1:
        impulse_sweep_set = 2*np.random.rand(sweep_initial, input_channels, history)-1
        # Bound sweep set to be within range of the original input data.
        for i in range(0, input_channels):
            min_value = input_range[i][0]
            max_value = input_range[i][1]
            impulse_sweep_set[:, i, :] = impulse_sweep_set[:, i, :]*(max_value-min_value)+min_value

    # Obtain the output for input impulses.
    print("Exciting model...")
    model.cpu()
    if sweep_initial != 1:
        impulse_response = np.zeros([combination_count, output_channels, sweep_initial])
    else:
        impulse_response = np.zeros([combination_count, output_channels, 1])
    batch_idx = 1
    batch_size_analyze = 256
    progress_bar = pyprind.ProgBar(len(range(0, combination_count, batch_size_analyze)), monitor=True)
    # Calculate the bias at the zero point.
    model_input = np.copy(impulse_array[0:1, :, :])
    bias = model(torch.tensor((model_input-mu_x)/sig_x, dtype=torch.float))*sig_y_t+mu_y_t

    # Calculate the response from all impulse combinations.
    for i in range(0, combination_count, batch_size_analyze):
        if i + batch_size_analyze > combination_count:
            # Handle the last batch.
            impulse = impulse_array[i:]
            if sweep_initial > 1:
                for j in range(0, sweep_initial):
                    mult = impulse_sweep_set[j, :, :]
                    model_input = mult*impulse*input_mask
                    output = (model(torch.tensor((model_input-mu_x)/sig_x, dtype=torch.float))*sig_y_t+mu_y_t).detach().cpu().numpy()
                    impulse_response[i:, :, j] = output
            else:
                model_input = impulse*input_mask
                output = (model(torch.tensor((model_input-mu_x)/sig_x, dtype=torch.float))*sig_y_t+mu_y_t).detach().cpu().numpy()
                impulse_response[i:, :, 0] = output
        else:
            # Handle a standard size batch.
            impulse = impulse_array[i:(i+batch_size_analyze)]
            if sweep_initial > 1:
                for j in range(0, sweep_initial):
                    mult = impulse_sweep_set[j, :, :]
                    model_input = mult*impulse*input_mask
                    output = (model(torch.tensor((model_input-mu_x)/sig_x, dtype=torch.float))*sig_y_t+mu_y_t).detach().cpu().numpy()
                    impulse_response[i:(i+batch_size_analyze), :, j] = output
            else:
                model_input = impulse*input_mask
                output = (model(torch.tensor((model_input-mu_x)/sig_x, dtype=torch.float))*sig_y_t+mu_y_t).detach().cpu().numpy()
                impulse_response[i:(i+batch_size_analyze), :, 0] = output
        batch_idx += 1
        progress_bar.update()
    #impulse_response = impulse_response.detach().numpy()
    time.sleep(0.5) # Allows progress bar to finish printing elapsed time.
    print()

    def process_subcombination(subcombination):
        sub_impulse = np.zeros([input_channels*history])
        # Determine index of combination in impulse_response
        for element in subcombination:
            sub_impulse[input_channels*history-1-element] = 1
        sub_index = array_to_int(sub_impulse.astype(int))
        # Loop through all subcombinations
        subsub_indices = []
        for l in range(0, len(subcombination)+1):
            for subsubcombination in itertools.combinations(subcombination, l):
                if subcombination != subsubcombination:
                    subsub_impulse = np.zeros([input_channels*history])
                    # Determine index of subcombination in impulse_response
                    for element in subsubcombination:
                        subsub_impulse[input_channels*history-1-element] = 1
                    subsub_index = array_to_int(subsub_impulse.astype(int))
                    subsub_indices.append(subsub_index)
        return sub_index, subsub_indices
    
    # Analyze responses (note: progress bar is not linear with computation time)
    print("Analyzing responses...")
    progress_bar = pyprind.ProgBar(combination_count, monitor=True)
    num_cores = multiprocessing.cpu_count()
    for combination_id in range(0, len(combinations)+1):
        # Loop all combinations
        results = Parallel(n_jobs=num_cores)(delayed(process_subcombination)(subcombination) \
                           for subcombination in itertools.combinations(combinations, combination_id))
        for each in results:
            sub_index = each[0]
            subsub_indices = each[1]
            for subsub_index in subsub_indices:
                impulse_response[sub_index, :, :] = impulse_response[sub_index, :, :] - \
                                                    impulse_response[subsub_index, :, :]
            progress_bar.update()
    time.sleep(0.5) # Allows progress bar to finish printing elapsed time.
    print()
    
    # Examine the impulse response for all combinations and generate a function.
    print("Estimating system equation...")
    # Create a mask of relevant inputs for later model retraining.
    new_mask = np.zeros([input_channels, history])
    # Create a sweep set for curve fitting.
    fit_sweep_set = np.random.rand(sweep_detailed, input_channels, history)
    for i in range(0, input_channels):
        min_value = input_range[i][0]
        max_value = input_range[i][1]
        fit_sweep_set[:, i, :] = fit_sweep_set[:, i, :]*(max_value-min_value)+min_value
    model_function = []
    for channel_id in range(0, output_channels):
        # Function for the output channel is a sum of product functions.
        channel_function = []
        # Get the magnitude average point value of each product function contribution.
        Z = np.sum(abs(impulse_response[:, channel_id, :]), 1)/sweep_initial
        # Get the variance of each product function.
        S = np.var(impulse_response[:, channel_id, :], 1)
        total_variance = sum(S)
        # Get indices of responses from largest to smallest.
        response_indices = np.flip(np.argsort(Z), 0)
        # Get indices of variances from largest to smallest.
        variance_indices = np.flip(np.argsort(S), 0)
        
        # Identify the top responses.
        if verbose:
            print("############################################################")
            print("Estimate of channel " + str(channel_id+1))
            print("############################################################")
        candidate_limit = min(25, len(response_indices))
        sig_indexes = []
        for k in range(0, candidate_limit):
            sig_index = response_indices[k]
            sig_response = Z[sig_index]
            z_sorted = np.flip(np.sort(Z[1:], 0), 0)
            contribution_magnitude = sig_response/sum(z_sorted)
            if contribution_magnitude > contrib_thresh:
                sig_indexes.append(sig_index)
        for k in range(0, candidate_limit):
            sig_index = variance_indices[k]
            sig_variance = S[sig_index]
            contribution_variance = sig_variance/total_variance
            if contribution_variance > contrib_thresh and sig_index not in sig_indexes:
                sig_indexes.append(sig_index)
                
        # Estimate equations for top responses.
        for sig_index in sig_indexes:
            sig_response = Z[sig_index]
            sig_variance = S[sig_index]
            sig_impulse = impulse_array[sig_index:sig_index+1, :, :]
            if verbose: print("Response ID " + str(sig_index) + " contribution:")
            
            # Process a product function if the response is significant.
            # Significance is % contribution to total magnitude or variance.
            # Bias is not included in magnitude significance.
            z_sorted = np.flip(np.sort(Z[1:], 0), 0)
            contribution_magnitude = sig_response/sum(z_sorted)
            contribution_variance = sig_variance/total_variance
            if sig_index is not 0:
                if verbose: print("Magnitude : " + str('%.1f'%(contribution_magnitude*100)) + "%")
                if verbose: print("Variance  : " + str('%.1f'%(contribution_variance*100)) + "%")
            else:
                if verbose: print("Bias contribution omitted from calculation.")
            if verbose: print("============================================================")
            if contribution_magnitude > contrib_thresh or contribution_variance > contrib_thresh:
    
                # Determine the arguments of the product function.
                arg_list = []
                for input_id in range(0, input_channels):
                    for element_id, element in enumerate(sig_impulse[0, input_id, :].astype(int)):
                        if element == 1:
                            delay = history - 1 - element_id
                            arg_list.append({"input_channel": input_id, "delay": delay})
                            new_mask[input_id, element_id] = 1
                
                # Create the product function template string.
                f_list = []
                f_str = "f("
                for _, arg in enumerate(arg_list):
                    f_list.append("x" + str(arg["input_channel"]+1) + "(k-" + str(arg["delay"]) + ")")
                for arg_num, arg_str in enumerate(f_list):
                    f_str = f_str + arg_str
                    if arg_num < len(f_list)-1:
                        f_str = f_str + ","
                if len(arg_list) == 0:
                    f_str = f_str + "0"
                f_str = f_str + ")"
            
                # Estimate the product function.
                def fcn_empty(_):
                    return 0
                def txt_empty(_):
                    return ""
                dct_empty = {
                    "txt": "?",
                    "txt_fcn": txt_empty,
                    "fcn": fcn_empty,
                    "upper": [],
                    "lower": [],
                    "weight": 1.0
                }
                product_function = {
                    "arg_list": arg_list,
                    "template_string": f_str,
                    "estimate_string": "f(?)",
                    "parameters": [],
                    "function": dct_empty,
                    "shift": []
                }
                if len(arg_list) > 0:
                    # Obtain sample points for curve fitting.
                    x_data = np.zeros([sweep_detailed, input_channels, history])
                    y_data = np.zeros([sweep_detailed, output_channels])
                    for idx in range(0, sweep_detailed):
                        mult = fit_sweep_set[idx, :, :]
                        model_input = mult*sig_impulse*input_mask
                        x_data[idx, :, :] = model_input
                        y_data[idx, :] = (model(torch.tensor((model_input-mu_x)/sig_x, dtype=torch.float))).detach().numpy()*sig_y+mu_y
                    # Recursively subtract contributions from product functions of arguments.
                    contribution_list = []
                    for idf in range(0, len(arg_list)):
                        new_contributions = []
                        for arg_combination in itertools.combinations(arg_list, idf):
                            arg_impulse = np.zeros([sweep_detailed, input_channels, history])
                            for arg in arg_combination:
                                arg_impulse[:, arg["input_channel"], history-1-arg["delay"]] = 1
                            model_input = arg_impulse * fit_sweep_set
                            output = (model(torch.tensor((model_input-mu_x)/sig_x, dtype=torch.float))).detach().numpy()*sig_y+mu_y
                            for contribution in contribution_list:
                                output = output - contribution
                            new_contributions.append(output)
                        contribution_list[0:0] = new_contributions
                    for contribution in contribution_list:
                        y_data = y_data - contribution
                    
                    # Format data for curve fitting
                    arg_count = len(arg_list)
                    x_data_fit = np.zeros([arg_count, sweep_detailed])
                    y_data_fit = np.zeros([sweep_detailed])
                    arg = 0
                    for i in range(0, input_channels):
                        for j in range(0, history):
                            if sig_impulse[0, i, j] == 1:
                                x_data_fit[arg, :] = x_data[:, i, j]
                                y_data_fit[:] = y_data[:, channel_id]
                                product_function["shift"].append(input_shift[i])
                                arg = arg + 1
                                
                    # Plot 2D and 3D data for visual inspection.
                    if save_visual == True or visual == True:
                        if arg_count == 1:
                            plt.figure()
                            plt.scatter(x_data_fit[0], y_data_fit, marker='.')
                            plt.title(product_function["template_string"])
                            plt.xlabel(f_list[0])
                            if save_visual == True:
                                plt.savefig('./output/analysis_{}/{}.pdf'.format(analysis_dir_count, \
                                            product_function["template_string"]))
                                pltDict = {"x": x_data_fit[0].tolist(),
                                       "y": y_data_fit.tolist()}
                                mat4py.savemat('./output/analysis_{}/{}.mat'.format(analysis_dir_count, \
                                               product_function["template_string"]), pltDict)
                            if visual == True: plt.show()
                        if arg_count == 2:
                            plt.figure()
                            ax = plt.axes(projection='3d')
                            ax.scatter3D(x_data_fit[0], x_data_fit[1], y_data_fit, c=y_data_fit, marker='o')
                            ax.set_title(product_function["template_string"])
                            ax.set_xlabel(f_list[0])
                            ax.set_ylabel(f_list[1])
                            if save_visual == True:
                                plt.savefig('./output/analysis_{}/{}.pdf'.format(analysis_dir_count, \
                                            product_function["template_string"]))
                                pltDict = {"x": x_data_fit[0].tolist(),
                                       "y": x_data_fit[1].tolist(),
                                       "z": y_data_fit.tolist()}
                                mat4py.savemat('./output/analysis_{}/{}.mat'.format(analysis_dir_count, \
                                               product_function["template_string"]), pltDict)
                            if visual == True: plt.show()
                    
                    # Estimate the product function using curve fitting.
                    if arg_count in functions:
                        candidate_functions = functions[arg_count]
                    else:
                        candidate_functions = []
                        product_function["estimate_string"] = product_function["template_string"]
                    best_fit = 100
                    for f in candidate_functions:
                        try:
                            popt, pcov = curve_fit(f["fcn"],
                                                   x_data_fit,
                                                   y_data_fit,
                                                   bounds=(f["lower"], f["upper"]),
                                                   maxfev=250000)
                            pcount = len(popt)
                            err = y_data_fit-f["fcn"](x_data_fit, *popt)
                            # Compute root mean squared error.
                            rmse = np.sqrt(sum(pow(err, 2))/sweep_detailed)
                            # Compute mean average error.
                            mae = np.mean(abs(err))
                            # Compute one standard deviation errors (just the normal std).
                            #std = np.sqrt(np.diag(pcov))
                            if verbose:
                                print("Fit for " + f["txt_fcn"](arg_list, product_function["shift"], *popt))
                                print("MAE  : " + str(FORMAT%mae))
                                print("RMSE : " + str(FORMAT%rmse))
                                #print("STD  : " + str(std))
                            f_weight = 1.0
                            if use_f_weight == True: f_weight = f["weight"]
                            if mae/f_weight < best_fit:
                                best_fit = mae/f_weight
                                product_function["parameters"] = popt
                                product_function["function"] = f
                                product_function["estimate_string"] = f["txt_fcn"](arg_list,
                                                                                   product_function["shift"],
                                                                                   *popt)
                                if verbose: print("Current best fit for Response " + str(sig_index))
                            if verbose: print()
                            # Perform curve fitting with different parameter initializations in attempt to improve fit.
                            fit_iterations = 5*pcount
                            for _ in range(1, fit_iterations):
                                pinit = np.random.rand(pcount)*(np.array(f["upper"])-np.array(f["lower"])) \
                                        + np.array(f["lower"])
                                popt_new, pcov = curve_fit(f["fcn"],
                                                           x_data_fit,
                                                           y_data_fit,
                                                           bounds=(f["lower"], f["upper"]),
                                                           p0=pinit,
                                                           maxfev=10000)
                                err = y_data_fit-f["fcn"](x_data_fit, *popt_new)
                                # Compute root mean squared error.
                                rmse = np.sqrt(sum(pow(err, 2))/sweep_detailed)
                                # Compute mean average error.
                                mae = np.mean(abs(err))
                                if mae/f_weight < 0.999*best_fit:
                                    best_fit = mae/f_weight
                                    product_function["parameters"] = popt_new
                                    product_function["function"] = f
                                    product_function["estimate_string"] = f["txt_fcn"](arg_list,
                                                                                       product_function["shift"],
                                                                                       *popt_new)
                                    if verbose:
                                        print("Revised fit for " + f["txt_fcn"](arg_list,
                                                                                product_function["shift"],
                                                                                *popt_new))
                                        print("MAE  : " + str(FORMAT%mae))
                                        print("RMSE : " + str(FORMAT%rmse))
                                        print("Current best fit for Response " + str(sig_index))
                                        print()
                        except Exception as e:
                            if best_fit == 100:
                                product_function["estimate_string"] = product_function["template_string"]
                            if verbose:
                                print("Warning: Fit could not be estimated for " + f["txt"] + ",")
                                print("         " + str(e))
                                print("")
                else:
                    # Handle constant bias at the zero point.
                    channel_bias = bias[0, channel_id].detach().numpy()
                    channel_bias_str = str('%.3f'%channel_bias)
                    product_function["parameters"] = [channel_bias]
                    def fcn_bias(x, a):
                        return a
                    def txt_bias(argList, argShift, a):
                        return str('%.3f'%a)
                    dct_bias = {
                        "txt": "a",
                        "fcn": fcn_bias,
                        "txt_fcn": txt_bias,
                        "upper": [2*channel_bias],
                        "lower": [0],
                        "weight": 1.0
                    }
                    product_function["function"] = dct_bias
                    product_function["estimate_string"] = channel_bias_str
                    if verbose:
                        print("Constant " + channel_bias_str)
                        print()
                    
                # Check if the candidate product function improves the accuracy of the model.
                if sig_index > 0:
                    current_function = [channel_function]
                    candidate_function = [channel_function + [product_function]]
                    current_metrics = evaluate_function(current_function,
                                                        input_data,
                                                        output_data[:, channel_id:channel_id+1])
                    candidate_metrics = evaluate_function(candidate_function,
                                                          input_data,
                                                          output_data[:, channel_id:channel_id+1])
                    
                    # Include product functions that are above a threshold and improve the overall MAE.
                    if candidate_metrics[0]["MAE"] > current_metrics[0]["MAE"]:
                        if verbose:
                            print("Warning: Candidate product function worsens overall MAE.")
                            print("         MAE increases from " + str(FORMAT%current_metrics[0]["MAE"])+\
                                  " to " + str(FORMAT%candidate_metrics[0]["MAE"]) + ".")
                        if contribution_magnitude < contrib_thresh_omit \
                        and contribution_variance < contrib_thresh_omit:
                            if verbose: print("         Candidate product function omitted.")
                        else:
                            channel_function.append(product_function)
                            if verbose: print("         Candidate product function added.")
                    else:
                        if verbose: print("Overall MAE declines from " + str(FORMAT%current_metrics[0]["MAE"]) \
                                          + " to " + str(FORMAT%candidate_metrics[0]["MAE"]) + ".")
                        channel_function.append(product_function)
                else:
                    channel_function.append(product_function)
            else:
                # Stop building the channel equation.
                if verbose:
                    print("Insignificant product function response.")
                    print()
                    print("############################################################")
                    print("Channel " + str(channel_id+1) + " function completed.")
                    print("############################################################")
                break
            if verbose: print()
        
        # Print the completed equation for the current output channel.     
        if verbose: print("System equation")
        if verbose: print("============================================================")
        # Print the function template for the current output channel.
        y_str = "y" + str(channel_id+1) + "[k] = "
        for idf, product_function in enumerate(channel_function):
            y_str = y_str + product_function["template_string"]
            if idf < len(channel_function) - 1:
                y_str = y_str + " + "
        print(y_str)
        y_str = "y" + str(channel_id+1) + "[k] = "
        for idf, product_function in enumerate(channel_function):
            if product_function["estimate_string"] != None:
                y_str = y_str + product_function["estimate_string"]
                if idf < len(channel_function) - 1:
                    y_str = y_str + " + "
        print(y_str)
        print()
        
        model_function.append(channel_function)
                
    return model_function, new_mask

# Future work: Use better fit metric than weighted MAE.
# https://autarkaw.org/2008/07/05/finding-the-optimum-polynomial-order-to-use-for-regression/
