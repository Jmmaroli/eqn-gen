# Estimate a discrete dynamical system equation for input_data and output_data.
# Author: John M. Maroli

import copy

from lib.create_model import create_model
from lib.analyze_model import analyze_model
from lib.evaluate_function import evaluate_function
from lib.tune_model import tune_model

FORMAT = '%.3e'

def estimate_equation(model_parameters, analysis_parameters, input_data, output_data):

    sweep_initial = analysis_parameters["sweep_initial"]
    sweep_detailed = analysis_parameters["sweep_detailed"]
    input_mask = 1
    
    # Initial round of training.
    print("Initial training and analysis")
    print("============================================================")
    model_dictionary_v1 = create_model(model_parameters, input_data, output_data, input_mask)

    model_function_v1, new_mask = analyze_model(analysis_parameters, model_dictionary_v1,
                                                input_data, output_data, input_mask)
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
    model_dictionary_v2 = create_model(model_parameters, input_data, output_data, new_mask)
    model_function_v2, _ = analyze_model(analysis_parameters, model_dictionary_v2,
                                         input_data, output_data, new_mask)
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
            model_function_v3.append(model_function_v2[c])
            metrics_v3.append(metrics_v2[c])
        else:
            print("Channel y" + str(c+1) + " did not improve")
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
                                                        input_data, output_data, input_mask)
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
            model_dictionary_v2 = create_model(model_parameters, input_data, output_data, new_mask)
            model_function_v2, _ = analyze_model(hf_analysis_parameters, model_dictionary_v2,
                                                 input_data, output_data, new_mask)
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
                    model_function_v3.append(model_function_v2[c])
                    metrics_v3.append(metrics_v2[c])
                else:
                    # Initial model results are better.
                    print("Channel y" + str(c+1) + " did not improve")
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
    
    print("Genetic algorithm tuning")
    print("============================================================")
    tuning_parameters = {"GA_population": analysis_parameters["GA_population"],
                         "GA_generations": analysis_parameters["GA_generations"],
                         "visual": analysis_parameters["visual"],
                         "save_visual": analysis_parameters["save_visual"]}
    model_function_v4 = tune_model(tuning_parameters, model_function_v3, input_data, output_data)
    metrics_v4 = evaluate_function(model_function_v4, input_data, output_data)
    
    model_function_v5 = []
    metrics_v5 = []
    for c in range(0,len(metrics_v4)):
        if metrics_v4[c]["MAE"] < metrics_v3[c]["MAE"]:
            print("Channel y" + str(c+1) + " improved")
            model_function_v5.append(model_function_v4[c])
            metrics_v5.append(metrics_v4[c])
        else:
            print("Channel y" + str(c+1) + " did not improve")
            model_function_v5.append(model_function_v3[c])
            metrics_v5.append(metrics_v3[c])
        print("MAE  : " + str(FORMAT%metrics_v3[c]["MAE"]) + " -> " + str(FORMAT%metrics_v4[c]["MAE"]))
        print("RMSE : " + str(FORMAT%metrics_v3[c]["RMSE"]) + " -> " + str(FORMAT%metrics_v4[c]["RMSE"]))
        
    print()
    
    print("Final estimation")
    print("============================================================")
    # Print GA tuned equations.
    for idc, channel_function in enumerate(model_function_v5):
        y_str = "y" + str(idc+1) + "[k] = "
        for idf, product_function in enumerate(channel_function):
            if product_function["estimate_string"] != None:
                y_str = y_str + product_function["estimate_string"]
                if idf < len(channel_function) - 1:
                    y_str = y_str + " + "
        print(y_str)
        print()
