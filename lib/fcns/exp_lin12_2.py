# Template product function: exp_lin12_2

import numpy as np

# Number of input variables
x_dim = 2

def fcn_exp_lin12_2(x,a):
    return a*x[0]*(np.exp(x[1])-1)

def txt_exp_lin12_2(argList,argShift,a):
    if all(shift == 0 for shift in argShift):
        return "{:.2e}*x{:d}[k-{:d}]*(e^(x{:d}[k-{:d}])-1)".format(
                a,argList[0]["input_channel"]+1,argList[0]["delay"],
                argList[1]["input_channel"]+1,argList[1]["delay"])
    else:
        return "{:.2e}*(x{:d}[k-{:d}]-{:.2e})*(e^(x{:d}[k-{:d}]-{:.2e})-1)".format(
                a,argList[0]["input_channel"]+1,argList[0]["delay"],argShift[0],
                argList[1]["input_channel"]+1,argList[1]["delay"],argShift[1])
        

dct_exp_lin12_2 = {
    "txt": "a*x1*(e^(x2)-1)",
    "txt_fcn": txt_exp_lin12_2,
    "fcn": fcn_exp_lin12_2,
    "upper": [10],
    "lower": [-10],
    "weight": 0.5
}

