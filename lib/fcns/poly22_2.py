# Template product function: poly22_2

import numpy as np

# Number of input variables
x_dim = 2

def fcn_poly22_2(x,a):
    return a*x[0]*x[1]

def txt_poly22_2(argList,argShift,a):
    if all(shift == 0 for shift in argShift):
        return "{:.2e}*x{:d}[k-{:d}]*x{:d}[k-{:d}]".format(
                a,argList[0]["input_channel"]+1,argList[0]["delay"],
                argList[1]["input_channel"]+1,argList[1]["delay"])
    else:
        return "{:.2e}*(x{:d}[k-{:d}]-{:.2e})*(x{:d}[k-{:d}]-{:.2e})".format(
                a,argList[0]["input_channel"]+1,argList[0]["delay"],argShift[0],
                argList[1]["input_channel"]+1,argList[1]["delay"],argShift[1])
        

dct_poly22_2 = {
    "txt": "a*x1*x2",
    "txt_fcn": txt_poly22_2,
    "fcn": fcn_poly22_2,
    "upper": [10],
    "lower": [-10],
    "weight": 1.0
}

