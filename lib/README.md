# Framework library

The framework is implemented as a single function:
```
estimate_equation(model_parameters, analysis_parameters, tuning_parameters, input_data, output_data)
```

This function combines library functions to generate a system equation representing the input-output data. The general steps towards system equation generation are presented as follows with their associated library function calls:

1\. Create a neural network model of the input-output data
```
model_dictionary = create_model(model_parameters, input_data, output_data, input_mask)
```

2\. Analyze the model through sensitivity analysis and fit template product functions to the excitation data to create an equation model
```
model_function, output_mask = analyze_model(analysis_parameters, model_dictionary, input_data, output_data, input_mask)
```

3\. Tune the equation model using a genetic algorithm
```
model_function_tuned = tune_model(tuning_parameters, model_function, input_data, output_data)
```

The framework utilizes the `output_mask` of the first call to `analyze_model()` to create an a more focused model, which is analyzed again and compared to the initial model before `tune_model()` is called. See `estimate_equation.py` for details.