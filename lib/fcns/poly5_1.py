# Template product function: poly5_1

import numpy as np

# Number of input variables
x_dim = 1

def fcn_poly5_1(x,a,b,c,d,e):
    return a*pow(x[0],5) \
         + b*pow(x[0],4) \
         + c*pow(x[0],3) \
         + d*pow(x[0],2) \
         + e*x[0]

def txt_poly5_1(argList,argShift,a,b,c,d,e):
    if all(shift == 0 for shift in argShift):
        return "{:.2e}*x{:d}[k-{:d}]^5 + " \
                "{:.2e}*x{:d}[k-{:d}]^4 + " \
                "{:.2e}*x{:d}[k-{:d}]^3 + " \
                "{:.2e}*x{:d}[k-{:d}]^2 + " \
                "{:.2e}*x{:d}[k-{:d}]".format(
                a,argList[0]["input_channel"]+1,argList[0]["delay"],
                b,argList[0]["input_channel"]+1,argList[0]["delay"],
                c,argList[0]["input_channel"]+1,argList[0]["delay"],
                d,argList[0]["input_channel"]+1,argList[0]["delay"],
                e,argList[0]["input_channel"]+1,argList[0]["delay"])
    else:
        return "{:.2e}*(x{:d}[k-{:d}]-{:.2e})^5 + " \
                "{:.2e}*(x{:d}[k-{:d}]-{:.2e})^4 + " \
                "{:.2e}*(x{:d}[k-{:d}]-{:.2e})^3 + " \
                "{:.2e}*(x{:d}[k-{:d}]-{:.2e})^2 + " \
                "{:.2e}*(x{:d}[k-{:d}]-{:.2e})".format(
                a,argList[0]["input_channel"]+1,argList[0]["delay"],argShift[0],
                b,argList[0]["input_channel"]+1,argList[0]["delay"],argShift[0],
                c,argList[0]["input_channel"]+1,argList[0]["delay"],argShift[0],
                d,argList[0]["input_channel"]+1,argList[0]["delay"],argShift[0],
                e,argList[0]["input_channel"]+1,argList[0]["delay"],argShift[0])
                

dct_poly5_1 = {
    "txt": "a*x1^5 + b*x1^4 + b*x1^3 + b*x1^2 + c*x1",
    "txt_fcn": txt_poly5_1,
    "fcn": fcn_poly5_1,
    "upper": [100, 100, 10, 10, 10],
    "lower": [-100, -100, -10, -10, -10],
    "weight": 0.10
}

