# Template product function: linear_4

import numpy as np

# Number of input variables
x_dim = 4

def fcn_linear_4(x,a):
    return a*x[0]*x[1]*x[2]*x[3]

def txt_linear_4(argList,argShift,a):
    if all(shift == 0 for shift in argShift):
        return "{:.3e}*x{:d}[k-{:d}]*x{:d}[k-{:d}]*x{:d}[k-{:d}]*x{:d}[k-{:d}]".format(
                a,argList[0]["input_channel"]+1,argList[0]["delay"],
                argList[1]["input_channel"]+1,argList[1]["delay"],
                argList[2]["input_channel"]+1,argList[2]["delay"],
                argList[3]["input_channel"]+1,argList[3]["delay"])
    else:
        return "{:.3e}*(x{:d}[k-{:d}]-{:.2e})*" \
               "(x{:d}[k-{:d}]-{:.2e})*" \
               "(x{:d}[k-{:d}]-{:.2e})*" \
               "(x{:d}[k-{:d}]-{:.2e})".format(
                a,argList[0]["input_channel"]+1,argList[0]["delay"],argShift[0],
                argList[1]["input_channel"]+1,argList[1]["delay"],argShift[1],
                argList[2]["input_channel"]+1,argList[2]["delay"],argShift[2],
                argList[3]["input_channel"]+1,argList[3]["delay"],argShift[3])
        

dct_linear_4 = {
    "txt": "a*x1*x2*x3*x4",
    "txt_fcn": txt_linear_4,
    "fcn": fcn_linear_4,
    "upper": [10],
    "lower": [-10],
    "weight": 1.0
}

