# Template product function: poly55_2

import numpy as np

# Number of input variables
x_dim = 2

def fcn_poly55_2(x,a,b,c,d,e,f,g,h,i,j):
    return a*pow(x[0],1)*pow(x[1],1) \
         + b*pow(x[0],2)*pow(x[1],1) \
         + c*pow(x[0],1)*pow(x[1],2) \
         + d*pow(x[0],3)*pow(x[1],1) \
         + e*pow(x[0],2)*pow(x[1],2) \
         + f*pow(x[0],1)*pow(x[1],3) \
         + g*pow(x[0],4)*pow(x[1],1) \
         + h*pow(x[0],3)*pow(x[1],2) \
         + i*pow(x[0],2)*pow(x[1],3) \
         + j*pow(x[0],1)*pow(x[1],4)

def txt_poly55_2(argList,argShift,a,b,c,d,e,f,g,h,i,j):
    if all(shift == 0 for shift in argShift):
        return "{:.2e}*x{:d}[k-{:d}]*x{:d}[k-{:d}] + " \
                "{:.2e}*x{:d}[k-{:d}]^2*x{:d}[k-{:d}] + " \
                "{:.2e}*x{:d}[k-{:d}]*x{:d}[k-{:d}]^2 + " \
                "{:.2e}*x{:d}[k-{:d}]^3*x{:d}[k-{:d}] + " \
                "{:.2e}*x{:d}[k-{:d}]^2*x{:d}[k-{:d}]^2 + " \
                "{:.2e}*x{:d}[k-{:d}]*x{:d}[k-{:d}]^3 + " \
                "{:.2e}*x{:d}[k-{:d}]^4*x{:d}[k-{:d}] + " \
                "{:.2e}*x{:d}[k-{:d}]^3*x{:d}[k-{:d}]^2 + " \
                "{:.2e}*x{:d}[k-{:d}]^2*x{:d}[k-{:d}]^3 + " \
                "{:.2e}*x{:d}[k-{:d}]*x{:d}[k-{:d}]^4".format(
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
                argList[1]["input_channel"]+1,argList[1]["delay"],
                g,argList[0]["input_channel"]+1,argList[0]["delay"],
                argList[1]["input_channel"]+1,argList[1]["delay"],
                h,argList[0]["input_channel"]+1,argList[0]["delay"],
                argList[1]["input_channel"]+1,argList[1]["delay"],
                i,argList[0]["input_channel"]+1,argList[0]["delay"],
                argList[1]["input_channel"]+1,argList[1]["delay"],
                j,argList[0]["input_channel"]+1,argList[0]["delay"],
                argList[1]["input_channel"]+1,argList[1]["delay"])
    else:
        return "{:.2e}*(x{:d}[k-{:d}]-{:.2e})*(x{:d}[k-{:d}]-{:.2e}) + " \
                "{:.2e}*(x{:d}[k-{:d}]-{:.2e})^2*(x{:d}[k-{:d}]-{:.2e}) + " \
                "{:.2e}*(x{:d}[k-{:d}]-{:.2e})*(x{:d}[k-{:d}]-{:.2e})^2 + " \
                "{:.2e}*(x{:d}[k-{:d}]-{:.2e})^3*(x{:d}[k-{:d}]-{:.2e}) + " \
                "{:.2e}*(x{:d}[k-{:d}]-{:.2e})^2*(x{:d}[k-{:d}]-{:.2e})^2 + " \
                "{:.2e}*(x{:d}[k-{:d}]-{:.2e})*(x{:d}[k-{:d}]-{:.2e})^3 + " \
                "{:.2e}*(x{:d}[k-{:d}]-{:.2e})^4*(x{:d}[k-{:d}]-{:.2e}) + " \
                "{:.2e}*(x{:d}[k-{:d}]-{:.2e})^3*(x{:d}[k-{:d}]-{:.2e})^2 + " \
                "{:.2e}*(x{:d}[k-{:d}]-{:.2e})^2*(x{:d}[k-{:d}]-{:.2e})^3 + " \
                "{:.2e}*(x{:d}[k-{:d}]-{:.2e})*(x{:d}[k-{:d}]-{:.2e})^4".format(
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
                argList[1]["input_channel"]+1,argList[1]["delay"],argShift[1],
                g,argList[0]["input_channel"]+1,argList[0]["delay"],argShift[0],
                argList[1]["input_channel"]+1,argList[1]["delay"],argShift[1],
                h,argList[0]["input_channel"]+1,argList[0]["delay"],argShift[0],
                argList[1]["input_channel"]+1,argList[1]["delay"],argShift[1],
                i,argList[0]["input_channel"]+1,argList[0]["delay"],argShift[0],
                argList[1]["input_channel"]+1,argList[1]["delay"],argShift[1],
                j,argList[0]["input_channel"]+1,argList[0]["delay"],argShift[0],
                argList[1]["input_channel"]+1,argList[1]["delay"],argShift[1])
                

dct_poly55_2 = {
    "txt": "a*x1*x2 + b*x1^2*x2 + c*x1*x2^2 + d*x1^3*x2 + e*x1^2*x2^2 + " \
    "f*x1*x2^3 + g*x1^4*x2 + h*x1^3*x2^2 + i*x1^2*x2^3 + j*x1*x2^4",
    "txt_fcn": txt_poly55_2,
    "fcn": fcn_poly55_2,
    "upper": [10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
    "lower": [-10, -10, -10, -10, -10, -10, -10, -10, -10, -10],
    "weight": 0.3
}

