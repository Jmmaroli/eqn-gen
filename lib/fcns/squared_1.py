# Template product function: squared_1

import numpy as np

# Number of input variables
x_dim = 1

def fcn_squared_1(x,a):
    return a*pow(x[0],2)

def txt_squared_1(argList,argShift,a):
    if all(shift == 0 for shift in argShift):
        return "{:.2e}*x{:d}[k-{:d}]^2".format(
                a,argList[0]["input_channel"]+1,argList[0]["delay"])
    else:
        return "{:.2e}*(x{:d}[k-{:d}]-{:.2e})^2".format(
                a,argList[0]["input_channel"]+1,argList[0]["delay"],argShift[0])


dct_squared_1 = {
    "txt": "a*x1^2",
    "txt_fcn": txt_squared_1,
    "fcn": fcn_squared_1,
    "upper": [10],
    "lower": [-10],
    "weight": 1.0
}

