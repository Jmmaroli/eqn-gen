# eqn-gen
A framework for generating discrete dynamical system equations from input-output data. A neural network model of the data is first created and then a form of sensitivity analysis is performed on the model to reimagine it as a sum of functions of its arguments. These component functions present a less complex identification problem and are estimated using curve fitting. The sum of component functions is the resultant system equation estimate. This process is integrated into an iterative framework to add robustness to the estimate.

System equations are generated as an input-output model
```
y[k] = f(y[k-1],...,y[k-n],u[k],...,u[k-n])
```

## Usage
The framework is implemented as a single function
```
estimate_equation(model_parameters, analysis_parameters, tuning_parameters, input_data, output_data)
```
This function takes a number of settings defined in the model_parameters, analysis_parameters, and tuning_parameters dictionaries. The input and output data are simply 2D numpy arrays in the form
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

## Installation

### Setting up a Virtual Environment

1. Create a virtual environment:
```bash
python -m venv venv
```

2. Activate the virtual environment:
   - On Linux/Mac:
   ```bash
   source venv/bin/activate
   ```
   - On Windows:
   ```bash
   venv\Scripts\activate
   ```

3. Install the package and all dependencies:
```bash
pip install -e .
```

Optional - install CUDA support for GPU
```bash
pip uninstall -y torch
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

This will install all required dependencies including:
* [PyTorch](https://github.com/pytorch/pytorch) - for learning input-output data
* [SciPy](https://www.scipy.org/) - for curve fitting
* [NumPy](https://numpy.org/) - for numerical computations
* [matplotlib](https://matplotlib.org/) - for plotting
* [PyPrind](https://github.com/rasbt/pyprind) - for progress indicators
* [h5py](https://www.h5py.org/) - for HDF5 data export
* [joblib](https://joblib.readthedocs.io/) - for parallel processing
* [psutil](https://github.com/giampaolo/psutil) - for system monitoring

## Prerequisites
* The [TCN](https://github.com/locuslab/TCN) model is a neural network variant used by the framework, however it is included within the code.
