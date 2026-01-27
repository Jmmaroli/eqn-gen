# Template product function: exp_1

import numpy as np

# Number of input variables
x_dim = 1

def fcn_exp_1(x,a,b):
    return a*(np.exp(b*x[0])-1)

def txt_exp_1(argList,argShift,a,b):
    if all(shift == 0 for shift in argShift):
        return "{:.2e}*(e^({:.2e}*x{:d}[k-{:d}])-1)".format(
                a,b,argList[0]["input_channel"]+1,argList[0]["delay"])
    else:
        return "{:.2e}*(e^({:.2e}*(x{:d}[k-{:d}]-{:.2e}))-1)".format(
                a,b,argList[0]["input_channel"]+1,argList[0]["delay"],argShift[0])
        

dct_exp_1 = {
    "txt": "a*(e^(b*x1)-1)",
    "txt_fcn": txt_exp_1,
    "fcn": fcn_exp_1,
    "upper": [10, 5],
    "lower": [-10, -5],
    "weight": 0.25
}

