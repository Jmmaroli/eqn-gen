# Template product function: linear_5

import numpy as np

# Number of input variables
x_dim = 5

def fcn_linear_5(x,a):
    return a*x[0]*x[1]*x[2]*x[3]*x[4]

def txt_linear_5(argList,argShift,a):
    if all(shift == 0 for shift in argShift):
        return "{:.3e}*x{:d}[k-{:d}]*x{:d}[k-{:d}]*x{:d}[k-{:d}]*x{:d}[k-{:d}]*x{:d}[k-{:d}]".format(
                a,argList[0]["input_channel"]+1,argList[0]["delay"],
                argList[1]["input_channel"]+1,argList[1]["delay"],
                argList[2]["input_channel"]+1,argList[2]["delay"],
                argList[3]["input_channel"]+1,argList[3]["delay"],
                argList[4]["input_channel"]+1,argList[4]["delay"])
    else:
        return "{:.3e}*(x{:d}[k-{:d}]-{:.2e})*" \
               "(x{:d}[k-{:d}]-{:.2e})*" \
               "(x{:d}[k-{:d}]-{:.2e})*" \
               "(x{:d}[k-{:d}]-{:.2e})*" \
               "(x{:d}[k-{:d}]-{:.2e})".format(
                a,argList[0]["input_channel"]+1,argList[0]["delay"],argShift[0],
                argList[1]["input_channel"]+1,argList[1]["delay"],argShift[1],
                argList[2]["input_channel"]+1,argList[2]["delay"],argShift[2],
                argList[3]["input_channel"]+1,argList[3]["delay"],argShift[3],
                argList[4]["input_channel"]+1,argList[4]["delay"],argShift[4])

dct_linear_5 = {
    "txt": "a*x1*x2*x3*x4*x5",
    "txt_fcn": txt_linear_5,
    "fcn": fcn_linear_5,
    "upper": [10],
    "lower": [-10],
    "weight": 1.0
}

