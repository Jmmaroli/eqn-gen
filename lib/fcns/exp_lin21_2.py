# Template product function: exp_lin21_2

import numpy as np

# Number of input variables
x_dim = 2

def fcn_exp_lin21_2(x,a):
    return a*x[1]*(np.exp(x[0])-1)

def txt_exp_lin21_2(argList,argShift,a):
    if all(shift == 0 for shift in argShift):
        return "{:.2e}*x{:d}[k-{:d}]*(e^(x{:d}[k-{:d}])-1)".format(
                a,argList[1]["input_channel"]+1,argList[1]["delay"],
                argList[0]["input_channel"]+1,argList[0]["delay"])
    else:
        return "{:.2e}*(x{:d}[k-{:d}]-{:.2e})*(e^(x{:d}[k-{:d}]-{:.2e})-1)".format(
                a,argList[1]["input_channel"]+1,argList[1]["delay"],argShift[1],
                argList[0]["input_channel"]+1,argList[0]["delay"],argShift[0])

dct_exp_lin21_2 = {
    "txt": "a*x2*(e^(x1)-1)",
    "txt_fcn": txt_exp_lin21_2,
    "fcn": fcn_exp_lin21_2,
    "upper": [10],
    "lower": [-10],
    "weight": 0.5
}

