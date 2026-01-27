# Template product function: poly44_2

import numpy as np

# Number of input variables
x_dim = 2

def fcn_poly44_2(x,a,b,c,d,e,f):
    return a*pow(x[0],1)*pow(x[1],1) \
         + b*pow(x[0],2)*pow(x[1],1) \
         + c*pow(x[0],1)*pow(x[1],2) \
         + d*pow(x[0],3)*pow(x[1],1) \
         + e*pow(x[0],2)*pow(x[1],2) \
         + f*pow(x[0],1)*pow(x[1],3)

def txt_poly44_2(argList,argShift,a,b,c,d,e,f):
    if all(shift == 0 for shift in argShift):
        return "{:.2e}*x{:d}[k-{:d}]*x{:d}[k-{:d}] + " \
                "{:.2e}*x{:d}[k-{:d}]^2*x{:d}[k-{:d}] + " \
                "{:.2e}*x{:d}[k-{:d}]*x{:d}[k-{:d}]^2 + " \
                "{:.2e}*x{:d}[k-{:d}]^3*x{:d}[k-{:d}] + " \
                "{:.2e}*x{:d}[k-{:d}]^2*x{:d}[k-{:d}]^2 + " \
                "{:.2e}*x{:d}[k-{:d}]*x{:d}[k-{:d}]^3".format(
                a,argList[0]["input_channel"]+1,argList[0]["delay"],
                argList[1]["input_channel"]+1,argList[1]["delay"],
                b,argList[0]["input_channel"]+1,argList[0]["delay"],
                argList[1]["input_channel"]+1,argList[1]["delay"],
                c,argList[0]["input_channel"]+1,argList[0]["delay"],
                argList[1]["input_channel"]+1,argList[1]["delay"],
                d,argList[0]["input_channel"]+1,argList[0]["delay"],
                argList[1]["input_channel"]+1,argList[1]["delay"],
                e,argList[0]["input_channel"]+1,argList[0]["delay"],
                argList[1]["input_channel"]+1,argList[1]["delay"],
                f,argList[0]["input_channel"]+1,argList[0]["delay"],
                argList[1]["input_channel"]+1,argList[1]["delay"])
    else:
        return "{:.2e}*(x{:d}[k-{:d}]-{:.2e})*(x{:d}[k-{:d}]-{:.2e}) + " \
                "{:.2e}*(x{:d}[k-{:d}]-{:.2e})^2*(x{:d}[k-{:d}]-{:.2e}) + " \
                "{:.2e}*(x{:d}[k-{:d}]-{:.2e})*(x{:d}[k-{:d}]-{:.2e})^2 + " \
                "{:.2e}*(x{:d}[k-{:d}]-{:.2e})^3*(x{:d}[k-{:d}]-{:.2e}) + " \
                "{:.2e}*(x{:d}[k-{:d}]-{:.2e})^2*(x{:d}[k-{:d}]-{:.2e})^2 + " \
                "{:.2e}*(x{:d}[k-{:d}]-{:.2e})*(x{:d}[k-{:d}]-{:.2e})^3".format(
                a,argList[0]["input_channel"]+1,argList[0]["delay"],argShift[0],
                argList[1]["input_channel"]+1,argList[1]["delay"],argShift[1],
                b,argList[0]["input_channel"]+1,argList[0]["delay"],argShift[0],
                argList[1]["input_channel"]+1,argList[1]["delay"],argShift[1],
                c,argList[0]["input_channel"]+1,argList[0]["delay"],argShift[0],
                argList[1]["input_channel"]+1,argList[1]["delay"],argShift[1],
                d,argList[0]["input_channel"]+1,argList[0]["delay"],argShift[0],
                argList[1]["input_channel"]+1,argList[1]["delay"],argShift[1],
                e,argList[0]["input_channel"]+1,argList[0]["delay"],argShift[0],
                argList[1]["input_channel"]+1,argList[1]["delay"],argShift[1],
                f,argList[0]["input_channel"]+1,argList[0]["delay"],argShift[0],
                argList[1]["input_channel"]+1,argList[1]["delay"],argShift[1])
        

dct_poly44_2 = {
    "txt": "a*x1*x2 + b*x1^2*x2 + c*x1*x2^2 + d*x1^3*x2 + e*x1^2*x2^2 + f*x1*x2^3",
    "txt_fcn": txt_poly44_2,
    "fcn": fcn_poly44_2,
    "upper": [10, 10, 10, 10, 10, 10],
    "lower": [-10, -10, -10, -10, -10, -10],
    "weight": 0.4
}

