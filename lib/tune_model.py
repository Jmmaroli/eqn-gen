# Tune model parameters using a genetic algorithm.

import random
import copy
import time
import os
import numpy as np
import pyprind
import matplotlib.pyplot as plt

from lib.evaluate_function import evaluate_function
from lib.format_channel_function import format_channel_function

def tune_model(tuning_parameters, model_function, input_data, output_data, output_dir=None, subfolder_name=None):
    
    population_size = tuning_parameters["ga_population"]
    generation_count = tuning_parameters["ga_generations"]
    visual = tuning_parameters["visual"]
    save_data = tuning_parameters["save_data"]
    seed = tuning_parameters["seed"]
    
    np.random.seed(seed)
    
    if save_data == True:
        if output_dir is not None and subfolder_name is not None:
            # Use the provided output directory and subfolder name
            ga_dir = os.path.join(output_dir, subfolder_name)
            os.makedirs(ga_dir, exist_ok=False)
        else:
            # Throw an exception if data is to be saved without specified directory
            raise ValueError('output_dir and subfolder_name must be specified to save data.')
    
    # Tune each channel individually.
    model_function_tuned = copy.deepcopy(model_function)
    for channel_id, channel_function in enumerate(model_function_tuned):
        
        # Get population details.
        parameter_count = 0
        upper_bounds = []
        lower_bounds = []
        parameters = []
        for fcn_id, product_function in enumerate(channel_function):
            parameter_count = parameter_count + len(product_function["parameters"])
            upper_bounds.extend(product_function["function"]["upper"])
            lower_bounds.extend(product_function["function"]["lower"])
            parameters.extend(product_function["parameters"])
            
        # Create the initial population of parameters.
        population = np.random.rand(population_size, parameter_count)
        # Member of initial estimate.
        for member_id in range(0, 1):
            population[member_id, :] = parameters
        # Members near initial estimate.
        for member_id in range(1, population_size):
            lower_bounds_dist = np.array(parameters) - (np.array(parameters) - np.array(lower_bounds))/3
            upper_bounds_dist = np.array(parameters) + (np.array(upper_bounds) - np.array(parameters))/3
            # Triangular distribution chosen because it is bounded.
            population[member_id, :] = np.random.triangular(lower_bounds_dist,parameters,upper_bounds_dist)
            
        # Execute genetic algorithm.
        print("Tuning channel " + str(channel_id+1) + "...")
        top_heuristic = np.zeros(generation_count)
        progress_bar = pyprind.ProgBar(generation_count, monitor=True)
        for generation_id in range(0, generation_count):
            # Evaluate generation.
            heuristic = np.zeros(population_size)
            for member_id in range(0, population_size):
                # Substitute the product function parameters
                parameter_index = 0
                for fcn_id, product_function in enumerate(channel_function):
                    product_function["parameters"] = list(
                            population[member_id, parameter_index:parameter_index+len(product_function["parameters"])])
                    parameter_index = parameter_index + len(product_function["parameters"])
                # Evaluate the new channel function
                metrics = evaluate_function([channel_function],
                                            input_data,
                                            output_data[:, channel_id:channel_id+1])
                heuristic[member_id]  = metrics[0]["MAE"]
            
            # Perform crossover of best members, clone best member.
            # Rank from smallest to largest MAE.
            member_rank = np.argsort(heuristic)
            upper_rank = member_rank[0:int(len(member_rank)/2)]
            population[0, :] = population[member_rank[0]]
            top_heuristic[generation_id] = heuristic[member_rank[0]]
            for member_id in range(1, population_size):
                parents = np.random.choice(list(upper_rank), size=2, replace=False)
                # Handle crossover only if there are multiple parameters.
                if parameter_count > 1:
                    crossover_point = np.random.randint(0, parameter_count + 1)
                    child = np.concatenate((population[parents[0], :crossover_point], population[parents[1], crossover_point:]))
                else:
                    # If one parameter, choose a parent.
                    child = population[parents[np.random.randint(0, 2)], :]
                population[member_id, :] = child
            
            # Perform mutations of new members.
            for member_id in range(1, population_size):
                if np.random.rand() < 0.25:
                    mutation_mask = np.random.randint(2, size=parameter_count)
                    mutation_degree = 0.1*2*(np.random.rand(parameter_count)-0.5)
                    mutation = mutation_mask*mutation_degree
                    population[member_id, :] = population[member_id, :] + mutation
                    # Enforce parameter bounds.
                    population[member_id, :] = np.clip(population[member_id, :], lower_bounds, upper_bounds)
            progress_bar.update()
        time.sleep(0.5) # Allows progress bar to finish printing elapsed time.

        # Assign new parameters to product function.
        parameter_index = 0
        for fcn_id, product_function in enumerate(channel_function):
            product_function["parameters"] = list(
                    population[0, parameter_index:parameter_index+len(product_function["parameters"])])
            if len(product_function["parameters"]) > 0:
                parameter_index = parameter_index + len(product_function["parameters"])
                product_function["estimate_string"] = product_function["function"]["txt_fcn"](
                        product_function["arg_list"],
                        product_function["shift"],
                        *product_function["parameters"])
        print()
        
        # Save tuned channel function to file if needed
        if save_data:
            # Format channel function strings
            y_str_template, y_str_estimate, y_str_mappings = format_channel_function(channel_function, channel_id+1)
            channel_str_detailed = y_str_template + '\n\n' + y_str_mappings + '\n\n' + y_str_estimate + '\n\n'
            
            # Save to tuned_equation.txt file
            equation_file_path = os.path.join(ga_dir, 'tuned_equation.txt')
            with open(equation_file_path, 'a') as f:
                f.write(channel_str_detailed)
        
        # Plot GA tuning metrics.
        if save_data == True or visual == True:
            plt.figure()
            plt.plot(top_heuristic)
            plt.title('Top MAE vs Generation')
            plt.xlabel('Generation')
            plt.ylabel('MAE')
            if save_data == True: plt.savefig(os.path.join(ga_dir, 'ga_mae.pdf'))
            if visual == True: plt.show()
        
    return model_function_tuned

# Future: In GA tuning, use a parameter confidence interval to limit the search space.
# http://kitchingroup.cheme.cmu.edu/blog/2013/02/12/Nonlinear-curve-fitting-with-parameter-confidence-intervals/
    