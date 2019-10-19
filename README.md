# eqn-gen
A framework for generating discrete dynamical system equations from input-output data. A neural network model of the data is first created and then a form of sensitivity analysis is performed on the model to reimagine it as a sum of functions of its arguments. These component functions present a less complex identification problem and are estimated using curve fitting. The sum of component functions is the resultant system equation estimate. This process is integrated into an iterative framework to add robustness to the estimate.

System equations are generated as an input-output model
```
y[k] = f(y[k-1],...,y[k-n],u[k],...,u[k-n])
```

## Prerequisites
* [PyTorch](https://github.com/pytorch/pytorch) is required for learning input-output data and must be installed.
* The [TCN](https://github.com/locuslab/TCN) model is a neural network variant used by the framework, however it is included within the code.
* Sometimes the analysis takes a very long time, so [Pyprind](https://github.com/rasbt/pyprind) is used to indicate progress.
* Data is exported in .mat format using [mat4py](https://pypi.org/project/mat4py/), however this can be removed if not needed.
* Curve fitting is performed using [SciPy](https://www.scipy.org/).

## Usage
The framework is implemented as a single function
```
estimate_equation(model_parameters, analysis_parameters, input_data, output_data)
```
This function takes a number of settings defined in the model_parameters and analysis_parameters dictionaries. The input and output data are simply 2D numpy arrays in the form
```
input_data  = |u1[k-N] u2[k-N] u3[k-N]   ...  |
              |  ...     ...     ...     ...  |
              | u1[k]   u2[k]   u3[k]    ...  |

output_data = |y1[k-N] y2[k-N] y3[k-N]   ...  |
              |  ...     ...     ...     ...  |
              | y1[k]   y2[k]   y3[k]    ...  |
```
where N is the number of samples. The data is simply arranged so that the last row is the most recent data and the first row is the initial data.

A script called run_example.py is provided with 10 examples. Examples are run using
```
python run_example.py #
```
where # indicates the example (indexed 0-9). This script contains everything needed to use the framework and is a simple starting point for analyzing data of your own.
