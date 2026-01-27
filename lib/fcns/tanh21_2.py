# Template product function: tanh21_2

import numpy as np

# Number of input variables
x_dim = 2

def fcn_tanh21_2(x,a,b,c,d,e):
    return a*np.tanh(b*x[1]-c)*np.tanh(d*x[0]-e) - a*np.tanh(-c)*np.tanh(-e)

def txt_tanh21_2(argList,argShift,a,b,c,d,e):
    if all(shift == 0 for shift in argShift):
        return "{:.2e}*tanh({:.2e}*x{:d}[k-{:d}]-{:.2e})*" \
                "tanh({:.2e}*x{:d}[k-{:d}]-{:.2e}) - " \
                "{:.2e}".format(
                        a,b,argList[1]["input_channel"]+1,argList[1]["delay"],c,
                        d,argList[0]["input_channel"]+1,argList[0]["delay"],e,
                        (a*np.tanh(-c)*np.tanh(-e)))
    else:
        return "{:.2e}*tanh({:.2e}*(x{:d}[k-{:d}]-{:.2e})-{:.2e})*" \
                "tanh({:.2e}*(x{:d}[k-{:d}]-{:.2e})-{:.2e}) - " \
                "{:.2e}".format(
                        a,b,argList[1]["input_channel"]+1,argList[1]["delay"],argShift[0],c,
                        d,argList[0]["input_channel"]+1,argList[0]["delay"],argShift[1],e,
                        (a*np.tanh(-c)*np.tanh(-e)))

dct_tanh21_2 = {
    "txt": "a*tanh(b*x2-c)*tanh(d*x1-e) - a*tanh(-c)*tanh(-e)",
    "txt_fcn": txt_tanh21_2,
    "fcn": fcn_tanh21_2,
    "upper": [1000, 1, 100, 1, 100],
    "lower": [-1000, 0.001, -100, 0.001, -100],
    "weight": 0.5
}

