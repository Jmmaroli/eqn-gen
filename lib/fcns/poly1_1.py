# Template product function: poly1_1

import numpy as np

# Number of input variables
x_dim = 1

def fcn_poly1_1(x,a):
    return a*x[0]

def txt_poly_1(argList,argShift,a):
    if all(shift == 0 for shift in argShift):
        return "{:.2e}*x{:d}[k-{:d}]".format(
                a,argList[0]["input_channel"]+1,argList[0]["delay"])
    else:
        return "{:.2e}*(x{:d}[k-{:d}]-{:.2e})".format(
                a,argList[0]["input_channel"]+1,argList[0]["delay"],argShift[0])
    

dct_poly1_1 = {
    "txt": "a*x1",
    "txt_fcn": txt_poly_1,
    "fcn": fcn_poly1_1,
    "upper": [10],
    "lower": [-10],
    "weight": 1.0
}

