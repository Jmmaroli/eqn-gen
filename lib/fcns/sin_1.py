# Template product function: sin_1

import numpy as np

# Number of input variables
x_dim = 1

def fcn_sin_1(x,a,b,c):
    return a*np.sin(b*x[0]+c)-a*np.sin(c)

def txt_sin_1(argList,argShift,a,b,c):
    if all(shift == 0 for shift in argShift):
        return "{:.2e}*sin({:.2e}*x{:d}[k-{:d}]+{:.2e})-{:.2e}".format(
                a,b,argList[0]["input_channel"]+1,argList[0]["delay"],c,(a*np.sin(c)))
    else:
        return "{:.2e}*sin({:.2e}*(x{:d}[k-{:d}]-{:.2e})+{:.2e})-{:.2e}".format(
                a,b,argList[0]["input_channel"]+1,argList[0]["delay"],argShift[0],c,(a*np.sin(c)))


dct_sin_1 = {
    "txt": "a*sin(b*x1+c) - a*sin(c)",
    "txt_fcn": txt_sin_1,
    "fcn": fcn_sin_1,
    "upper": [10, 10*3.14159, 3.14159],
    "lower": [0, 0, -3.14159],
    "weight": 0.25
}

