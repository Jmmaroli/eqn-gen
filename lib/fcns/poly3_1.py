# Template product function: poly3_1

import numpy as np

# Number of input variables
x_dim = 1

def fcn_poly3_1(x,a,b,c):
    return a*pow(x[0],3) \
         + b*pow(x[0],2) \
         + c*x[0]

def txt_poly3_1(argList,argShift,a,b,c):
    if all(shift == 0 for shift in argShift):
        return "{:.2e}*x{:d}[k-{:d}]^3 + " \
                "{:.2e}*x{:d}[k-{:d}]^2 + " \
                "{:.2e}*x{:d}[k-{:d}]".format(
                a,argList[0]["input_channel"]+1,argList[0]["delay"],
                b,argList[0]["input_channel"]+1,argList[0]["delay"],
                c,argList[0]["input_channel"]+1,argList[0]["delay"])
    else:
        return "{:.2e}*(x{:d}[k-{:d}]-{:.2e})^3 + " \
                "{:.2e}*(x{:d}[k-{:d}]-{:.2e})^2 + " \
                "{:.2e}*(x{:d}[k-{:d}]-{:.2e})".format(
                a,argList[0]["input_channel"]+1,argList[0]["delay"],argShift[0],
                b,argList[0]["input_channel"]+1,argList[0]["delay"],argShift[0],
                c,argList[0]["input_channel"]+1,argList[0]["delay"],argShift[0])
                

dct_poly3_1 = {
    "txt": "a*x1^3 + b*x1^2 + c*x1",
    "txt_fcn": txt_poly3_1,
    "fcn": fcn_poly3_1,
    "upper": [10, 10, 10],
    "lower": [-10, -10, -10],
    "weight": 0.20
}

