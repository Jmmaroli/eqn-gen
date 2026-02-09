# Estimate a discrete dynamical system equation for input_data and output_data.
# Author: John M. Maroli

import copy
import os
import numpy as np

from lib.create_model import create_model
from lib.analyze_model import analyze_model
from lib.evaluate_function import evaluate_function
from lib.tune_model import tune_model
from lib.format_channel_function import format_channel_function

FORMAT = '%.3e'

def estimate_equation(model_parameters, analysis_parameters, tuning_parameters, input_data, output_data):

    sweep_initial = analysis_parameters["sweep_initial"]
    sweep_detailed = analysis_parameters["sweep_detailed"]
    input_mask = 1
    
    # Create the parent estimate folder
    if not os.path.exists('./output'):
        os.mkdir('./output')
    estimate_dir_count = 1
    while os.path.exists('./output/estimate_{}'.format(estimate_dir_count)):
        estimate_dir_count = estimate_dir_count + 1
    estimate_dir = './output/estimate_{}'.format(estimate_dir_count)
    os.mkdir(estimate_dir)
    
    # Initial round of training.
    print("Initial training and analysis")
    print("============================================================")
    model_dictionary_v1 = create_model(model_parameters, input_data, output_data, input_mask, 
                                      output_dir=estimate_dir, subfolder_name='model_initial')

    model_function_v1, new_mask = analyze_model(analysis_parameters, model_dictionary_v1,
                                                input_data, output_data, input_mask,
                                                output_dir=estimate_dir, subfolder_name='analysis_initial')
    metrics_v1 = evaluate_function(model_function_v1, input_data, output_data)
    
    for channel_id, channel_metrics in enumerate(metrics_v1):
        print("Channel y" + str(channel_id+1) + " metrics")
        print("MAE  : " + str(FORMAT%channel_metrics["MAE"]))
        print("RMSE : " + str(FORMAT%channel_metrics["RMSE"]))
        print("MAX  : " + str(FORMAT%channel_metrics["MAX"]))
        print("MIN  : " + str(FORMAT%channel_metrics["MIN"]))
        print()
    
    # Model retraining after first analysis.
    print("Masked retraining and analysis")
    print("============================================================")
    print("New mask")
    print(new_mask)
    print()
    
    if (np.all(new_mask == 1)):
        # Skip retraining if mask is all 1s
        model_function_v3 = model_function_v1
        metrics_v3 = metrics_v1
        print("Mask is all 1s, skipping masked retraining and analysis.")
    else:
        model_dictionary_v2 = create_model(model_parameters, input_data, output_data, new_mask,
                                          output_dir=estimate_dir, subfolder_name='model_masked')
        model_function_v2, _ = analyze_model(analysis_parameters, model_dictionary_v2,
                                             input_data, output_data, new_mask,
                                             output_dir=estimate_dir, subfolder_name='analysis_masked')
        metrics_v2 = evaluate_function(model_function_v2, input_data, output_data)
        
        for channel_id, channel_metrics in enumerate(metrics_v2):
            print("Channel y" + str(channel_id+1) + " metrics")
            print("MAE  : " + str(FORMAT%channel_metrics["MAE"]))
            print("RMSE : " + str(FORMAT%channel_metrics["RMSE"]))
            print("MAX  : " + str(FORMAT%channel_metrics["MAX"]))
            print("MIN  : " + str(FORMAT%channel_metrics["MIN"]))
            print()
            
        model_function_v3 = []
        metrics_v3 = []
        for c in range(0, len(metrics_v2)):
            if metrics_v2[c]["MAE"] < metrics_v1[c]["MAE"]:
                print("Channel y" + str(c+1) + " improved")
                print("Continuing with function from masked model")
                model_function_v3.append(model_function_v2[c])
                metrics_v3.append(metrics_v2[c])
            else:
                print("Channel y" + str(c+1) + " did not improve")
                print("Continuing with function from previous model")
                model_function_v3.append(model_function_v1[c])
                metrics_v3.append(metrics_v1[c])
            print("MAE  : " + str(FORMAT%metrics_v1[c]["MAE"]) + " -> " + str(FORMAT%metrics_v2[c]["MAE"]))
            print("RMSE : " + str(FORMAT%metrics_v1[c]["RMSE"]) + " -> " + str(FORMAT%metrics_v2[c]["RMSE"]))
    print()
    
    # Check if data fits the model and re-analyze if needed.
    mae_percent_range = []
    for c in range(0, len(metrics_v3)):
        mae_percent_range.append(metrics_v3[c]["MAE"]/(metrics_v3[c]["MAX"]-metrics_v3[c]["MIN"]))
    # Threshold value for deeper analysis as MAE percent of total range.
    if any(value > 0.10 for value in mae_percent_range):
        hf_loop_limit = 5
        hf_loop_count = 1
        while hf_loop_count <= hf_loop_limit:
            hf_sweep_initial = sweep_initial*(5*hf_loop_count)
            hf_sweep_detailed = sweep_detailed*(int(0.5*hf_loop_count)+1)
            
            hf_analysis_parameters = copy.deepcopy(analysis_parameters)
            hf_analysis_parameters["sweep_initial"] = hf_sweep_initial
            hf_analysis_parameters["sweep_detailed"] = hf_sweep_detailed
            
            # Perform higher fidelity analysis on initial model.
            print("############################################################")
            print("Poor fit: higher fidelity analysis required")
            print("Increasing initial sweep multiples from {:d} to {:d}".format(sweep_initial, hf_sweep_initial))
            print("############################################################")
            print()
            print("High fidelity analysis iteration " + str(hf_loop_count))
            print("============================================================")
            model_function_v1, new_mask = analyze_model(hf_analysis_parameters, model_dictionary_v1,
                                                        input_data, output_data, input_mask,
                                                        output_dir=estimate_dir, subfolder_name=f'hf{hf_loop_count}_analysis_initial')
            metrics_v1 = evaluate_function(model_function_v1, input_data, output_data)
            
            for channel_id, channel_metrics in enumerate(metrics_v1):
                print("Channel y" + str(channel_id+1) + " metrics")
                print("MAE  : " + str(FORMAT%channel_metrics["MAE"]))
                print("RMSE : " + str(FORMAT%channel_metrics["RMSE"]))
                print("MAX  : " + str(FORMAT%channel_metrics["MAX"]))
                print("MIN  : " + str(FORMAT%channel_metrics["MIN"]))
                print()
        
            # Perform higher fidelity analysis on retrained model.
            print("Masked retraining and analysis")
            print("============================================================")
            print("New mask")
            print(new_mask)
            print()
            
            if (np.all(new_mask == 1)):
                # Skip retraining if mask is all 1s
                model_function_v3 = model_function_v1
                metrics_v3 = metrics_v1
                print("Mask is all 1s, skipping masked retraining and analysis.")
            else:
                model_dictionary_v2 = create_model(model_parameters, input_data, output_data, new_mask,
                                                  output_dir=estimate_dir, subfolder_name=f'hf{hf_loop_count}_model_masked')
                model_function_v2, _ = analyze_model(hf_analysis_parameters, model_dictionary_v2,
                                                     input_data, output_data, new_mask,
                                                     output_dir=estimate_dir, subfolder_name=f'hf{hf_loop_count}_analysis_masked')
                metrics_v2 = evaluate_function(model_function_v2, input_data, output_data)
            
                for channel_id, channel_metrics in enumerate(metrics_v2):
                    print("Channel y" + str(channel_id+1) + " metrics")
                    print("MAE  : " + str(FORMAT%channel_metrics["MAE"]))
                    print("RMSE : " + str(FORMAT%channel_metrics["RMSE"]))
                    print("MAX  : " + str(FORMAT%channel_metrics["MAX"]))
                    print("MIN  : " + str(FORMAT%channel_metrics["MIN"]))
                    print()
            
                # Combined analysis evaluation.
                model_function_v3 = []
                metrics_v3 = []
                for c in range(0, len(metrics_v2)):
                    if metrics_v2[c]["MAE"] < metrics_v1[c]["MAE"]:
                        # Retrained model results are better.
                        print("Channel y" + str(c+1) + " improved")
                        print("Continuing with function from masked model")
                        model_function_v3.append(model_function_v2[c])
                        metrics_v3.append(metrics_v2[c])
                    else:
                        # Initial model results are better.
                        print("Channel y" + str(c+1) + " did not improve")
                        print("Continuing with function from previous model")
                        model_function_v3.append(model_function_v1[c])
                        metrics_v3.append(metrics_v1[c])
                    print("MAE  : " + str(FORMAT%metrics_v1[c]["MAE"]) + " -> " + str(FORMAT%metrics_v2[c]["MAE"]))
                    print("RMSE : " + str(FORMAT%metrics_v1[c]["RMSE"]) + " -> " + str(FORMAT%metrics_v2[c]["RMSE"]))
            print()
                    
            # Check if the data fits the model.
            mae_percent_range = []
            for c in range(0, len(metrics_v3)):
                mae_percent_range.append(metrics_v3[c]["MAE"]/(metrics_v3[c]["MAX"]-metrics_v3[c]["MIN"]))
            if all(value < 0.10 for value in mae_percent_range):
                print("Higher fidelity analysis was successful")
                break
        
            # Set a limit on depth of analysis.
            hf_loop_count = hf_loop_count + 1
            if hf_loop_count == hf_loop_limit:
                print("WARNING: Higher fidelity analysis failed to fit data,")
                print("         parameter tuning will still be attempted")
                break
        print()
    else:
        print("Model analysis sufficient, skipping high fidelity analysis.")
        print()
    
    print("Genetic algorithm tuning")
    print("============================================================")
    model_function_v4 = tune_model(tuning_parameters, model_function_v3, input_data, output_data, 
                                   output_dir=estimate_dir, subfolder_name='ga_tuning')
    metrics_v4 = evaluate_function(model_function_v4, input_data, output_data)
    
    # Examine the GA tuned equation for each channel for improvement.
    model_function_v5 = []
    metrics_v5 = []
    for c in range(0,len(metrics_v4)):
        if metrics_v4[c]["MAE"] < metrics_v3[c]["MAE"]:
            print("Channel y" + str(c+1) + " improved")
            print("Using GA-tuned function")
            model_function_v5.append(model_function_v4[c])
            metrics_v5.append(metrics_v4[c])
        else:
            print("Channel y" + str(c+1) + " did not improve")
            print("Using untuned function")
            model_function_v5.append(model_function_v3[c])
            metrics_v5.append(metrics_v3[c])
        print("MAE  : " + str(FORMAT%metrics_v3[c]["MAE"]) + " -> " + str(FORMAT%metrics_v4[c]["MAE"]))
        print("RMSE : " + str(FORMAT%metrics_v3[c]["RMSE"]) + " -> " + str(FORMAT%metrics_v4[c]["RMSE"]))
        
    print()
    
    print("Final estimation")
    print("============================================================")
    
    # Save and print final system equations
    final_equation_path = os.path.join(estimate_dir, 'final_equation.txt')
    with open(final_equation_path, 'w') as f:
        for channel_id, channel_function in enumerate(model_function_v5):
            # Format channel function strings
            y_str_template, y_str_estimate, y_str_mappings = format_channel_function(channel_function, channel_id+1)
            channel_str_detailed = y_str_template + '\n\n' + y_str_mappings + '\n\n' + y_str_estimate + '\n\n'
            
            # Save to file
            f.write(channel_str_detailed)
            
            # Print to console
            print(y_str_estimate)
            print()
