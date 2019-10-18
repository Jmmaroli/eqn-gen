# Evaluate the function generated by model analysis.

import numpy as np

def evaluate_function(model_function, input_data, output_data):
    
    metrics = []
    
    data_length = input_data.shape[0]
    output_channels = output_data.shape[1]
    
    #for channel in range(0, len(model_function)):
    for channel_id, channel_function in enumerate(model_function):
        
        channel_y = np.zeros([data_length, output_channels])
        max_delay_overall = 0
        
        for _, product_function in enumerate(channel_function):
            
            arg_list = product_function["arg_list"]
            arg_count = len(arg_list)
            arg_shift = product_function["shift"]
            if arg_count > 0:
                
                max_delay = max([arg["delay"] for arg in arg_list])
                if max_delay > max_delay_overall:
                    max_delay_overall = max_delay
                
                # Arrange input data to be fed into product function.
                x = np.zeros([arg_count, data_length-max_delay])
                for i, arg in enumerate(arg_list):
                    input_channel = arg["input_channel"]
                    delay = arg["delay"]
                    if delay > 0:
                        x[i] = input_data[(max_delay-delay):-delay, input_channel]-arg_shift[i]
                    else:
                        x[i] = input_data[(max_delay-delay):, input_channel]-arg_shift[i]
                        
                # Feed input data to product function.
                params = product_function["parameters"]
                y_est = product_function["function"]["fcn"](x, *params)
                channel_y[max_delay:, channel_id] = channel_y[max_delay:, channel_id] + y_est
            else:
                # No arguments = bias term.
                params = product_function["parameters"]
                bias = product_function["function"]["fcn"](_, *params)
                channel_y[:, channel_id] = channel_y[:, channel_id] + bias
                
        # Calculate the error. 
        y_err = output_data[max_delay_overall:, channel_id] - channel_y[max_delay_overall:, channel_id]
        y_err = y_err[max_delay_overall+1:]
        mae = np.mean(abs(y_err))
        rmse = np.sqrt(sum(pow(y_err, 2))/len(y_err))
        ymax = np.max(output_data[max_delay_overall:, channel_id])
        ymin = np.min(output_data[max_delay_overall:, channel_id])
        
        channel_metrics = {
            "MAE": mae,
            "RMSE": rmse,
            "MAX": ymax,
            "MIN": ymin,
        }
        metrics.append(channel_metrics)
        
    return metrics

# Future work: examine MAE vs RMSE
# https://medium.com/human-in-a-machine-world/mae-and-rmse-which-metric-is-better-e60ac3bde13d