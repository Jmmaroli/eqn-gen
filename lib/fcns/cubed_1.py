# Template product function: cubed_1

import numpy as np

# Number of input variables
x_dim = 1

def fcn_cubed_1(x,a):
    return a*pow(x[0],3)

def txt_cubed_1(argList,argShift,a):
    if all(shift == 0 for shift in argShift):
        return "{:.2e}*x{:d}[k-{:d}]^3".format(
                a,argList[0]["input_channel"]+1,argList[0]["delay"])
    else:
        return "{:.2e}*(x{:d}[k-{:d}]-{:.2e})^3".format(
                a,argList[0]["input_channel"]+1,argList[0]["delay"],argShift[0])
        

dct_cubed_1 = {
    "txt": "a*x1^3",
    "txt_fcn": txt_cubed_1,
    "fcn": fcn_cubed_1,
    "upper": [10],
    "lower": [-10],
    "weight": 1.0
}

