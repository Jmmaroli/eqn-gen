# Template product function: tanhx_1

import numpy as np

# Number of input variables
x_dim = 1

def fcn_tanhx_1(x,a,b,c,d):
    return a*np.tanh(b*(x[0]+c)) + d*(x[0]+c) - (a*np.tanh(b*c)+d*c)

def txt_tanhx_1(argList,argShift,a,b,c,d):
    if all(shift == 0 for shift in argShift):
        return "{:.2e}*tanh({:.2e}*(x{:d}[k-{:d}]+{:.2e})) + " \
                "{:.2e}*(x{:d}[k-{:d}]+{:.2e}) - " \
                "{:.2e}".format(
                a,b,argList[0]["input_channel"]+1,argList[0]["delay"],c,
                d,argList[0]["input_channel"]+1,argList[0]["delay"],c,
                (a*np.tanh(b*c)+d*c))
    else:
        return "{:.2e}*tanh({:.2e}*((x{:d}[k-{:d}]-{:.2e})+{:.2e})) + " \
                "{:.2e}*((x{:d}[k-{:d}]-{:.2e})+{:.2e}) - " \
                "{:.2e}".format(
                a,b,argList[0]["input_channel"]+1,argList[0]["delay"],argShift[0],c,
                d,argList[0]["input_channel"]+1,argList[0]["delay"],argShift[0],c,
                (a*np.tanh(b*c)+d*c))


dct_tanhx_1 = {
    "txt": "a*tanh(b*(x1+c)) + d*(x1+c) - (a*tanh(b*c)+d*c)",
    "txt_fcn": txt_tanhx_1,
    "fcn": fcn_tanhx_1,
    "upper": [1000, 100, 100, 100],
    "lower": [-1000, 0.001, -100, -100],
    "weight": 0.25
}

