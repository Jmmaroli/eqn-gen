# Template product function: tanh_1

import numpy as np

# Number of input variables
x_dim = 1

def fcn_tanh_1(x,a,b):
    return a*np.tanh(b*x[0])

def txt_tanh_1(argList,argShift,a,b):
    if all(shift == 0 for shift in argShift):
        return "{:.2e}*tanh({:.2e}*x{:d}[k-{:d}])".format(
                a,b,argList[0]["input_channel"]+1,argList[0]["delay"])
    else:
        return "{:.2e}*tanh({:.2e}*(x{:d}[k-{:d}]-{:.2e}))".format(
                a,b,argList[0]["input_channel"]+1,argList[0]["delay"],argShift[0])


dct_tanh_1 = {
    "txt": "a*tanh(b*x1)",
    "txt_fcn": txt_tanh_1,
    "fcn": fcn_tanh_1,
    "upper": [1000, 10],
    "lower": [-1000, 0.001],
    "weight": 0.25
}

