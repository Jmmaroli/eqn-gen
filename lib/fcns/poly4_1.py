# Template product function: poly4_1

import numpy as np

# Number of input variables
x_dim = 1

def fcn_poly4_1(x,a,b,c,d):
    return a*pow(x[0],4) \
         + b*pow(x[0],3) \
         + c*pow(x[0],2) \
         + d*x[0]

def txt_poly4_1(argList,argShift,a,b,c,d):
    if all(shift == 0 for shift in argShift):
        return "{:.2e}*x{:d}[k-{:d}]^4 + " \
                "{:.2e}*x{:d}[k-{:d}]^3 + " \
                "{:.2e}*x{:d}[k-{:d}]^2 + " \
                "{:.2e}*x{:d}[k-{:d}]".format(
                a,argList[0]["input_channel"]+1,argList[0]["delay"],
                b,argList[0]["input_channel"]+1,argList[0]["delay"],
                c,argList[0]["input_channel"]+1,argList[0]["delay"],
                d,argList[0]["input_channel"]+1,argList[0]["delay"])
    else:
        return "{:.2e}*(x{:d}[k-{:d}]-{:.2e})^4 + " \
                "{:.2e}*(x{:d}[k-{:d}]-{:.2e})^3 + " \
                "{:.2e}*(x{:d}[k-{:d}]-{:.2e})^2 + " \
                "{:.2e}*(x{:d}[k-{:d}]-{:.2e})".format(
                a,argList[0]["input_channel"]+1,argList[0]["delay"],argShift[0],
                b,argList[0]["input_channel"]+1,argList[0]["delay"],argShift[0],
                c,argList[0]["input_channel"]+1,argList[0]["delay"],argShift[0],
                d,argList[0]["input_channel"]+1,argList[0]["delay"],argShift[0])
                

dct_poly4_1 = {
    "txt": "a*x1^4 + b*x1^3 + b*x1^2 + c*x1",
    "txt_fcn": txt_poly4_1,
    "fcn": fcn_poly4_1,
    "upper": [100, 100, 10, 10],
    "lower": [-100, -100, -10, -10],
    "weight": 0.15
}

