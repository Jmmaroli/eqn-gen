# Define the template product functions.

import numpy as np

# Polynomials.
#=============================================================================#
def fcn_poly1_1(x,a):
    return a*x[0]
def txt_poly_1(argList,argShift,a):
    if all(shift == 0 for shift in argShift):
        return "{:.2e}*x{:d}[k-{:d}]".format(
                a,argList[0]["input_channel"]+1,argList[0]["delay"])
    else:
        return "{:.2e}*(x{:d}[k-{:d}]-{:.2e})".format(
                a,argList[0]["input_channel"]+1,argList[0]["delay"],argShift[0])
    
def fcn_poly2_1(x,a,b):
    return a*pow(x[0],2) \
         + b*x[0]
def txt_poly2_1(argList,argShift,a,b):
    if all(shift == 0 for shift in argShift):
        return "{:.2e}*x{:d}[k-{:d}]^2 + " \
                "{:.2e}*x{:d}[k-{:d}]".format(
                a,argList[0]["input_channel"]+1,argList[0]["delay"],
                b,argList[0]["input_channel"]+1,argList[0]["delay"])
    else:
        return "{:.2e}*(x{:d}[k-{:d}]-{:.2e})^2 + " \
                "{:.2e}*(x{:d}[k-{:d}]-{:.2e})".format(
                a,argList[0]["input_channel"]+1,argList[0]["delay"],argShift[0],
                b,argList[0]["input_channel"]+1,argList[0]["delay"],argShift[0])

def fcn_squared_1(x,a):
    return a*pow(x[0],2)
def txt_squared_1(argList,argShift,a):
    if all(shift == 0 for shift in argShift):
        return "{:.2e}*x{:d}[k-{:d}]^2".format(
                a,argList[0]["input_channel"]+1,argList[0]["delay"])
    else:
        return "{:.2e}*(x{:d}[k-{:d}]-{:.2e})^2".format(
                a,argList[0]["input_channel"]+1,argList[0]["delay"],argShift[0])

def fcn_poly3_1(x,a,b,c):
    return a*pow(x[0],3) \
         + b*pow(x[0],2) \
         + c*x[0]
def txt_poly3_1(argList,argShift,a,b,c):
    if all(shift == 0 for shift in argShift):
        return "{:.2e}*x{:d}[k-{:d}]^3 + " \
                "{:.2e}*x{:d}[k-{:d}]^2 + " \
                "{:.2e}*x{:d}[k-{:d}]".format(
                a,argList[0]["input_channel"]+1,argList[0]["delay"],
                b,argList[0]["input_channel"]+1,argList[0]["delay"],
                c,argList[0]["input_channel"]+1,argList[0]["delay"])
    else:
        return "{:.2e}*(x{:d}[k-{:d}]-{:.2e})^3 + " \
                "{:.2e}*(x{:d}[k-{:d}]-{:.2e})^2 + " \
                "{:.2e}*(x{:d}[k-{:d}]-{:.2e})".format(
                a,argList[0]["input_channel"]+1,argList[0]["delay"],argShift[0],
                b,argList[0]["input_channel"]+1,argList[0]["delay"],argShift[0],
                c,argList[0]["input_channel"]+1,argList[0]["delay"],argShift[0])
                
def fcn_cubed_1(x,a):
    return a*pow(x[0],3)
def txt_cubed_1(argList,argShift,a):
    if all(shift == 0 for shift in argShift):
        return "{:.2e}*x{:d}[k-{:d}]^3".format(
                a,argList[0]["input_channel"]+1,argList[0]["delay"])
    else:
        return "{:.2e}*(x{:d}[k-{:d}]-{:.2e})^3".format(
                a,argList[0]["input_channel"]+1,argList[0]["delay"],argShift[0])
        
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
                
def fcn_poly22_2(x,a):
    return a*x[0]*x[1]
def txt_poly22_2(argList,argShift,a):
    if all(shift == 0 for shift in argShift):
        return "{:.2e}*x{:d}[k-{:d}]*x{:d}[k-{:d}]".format(
                a,argList[0]["input_channel"]+1,argList[0]["delay"],
                argList[1]["input_channel"]+1,argList[1]["delay"])
    else:
        return "{:.2e}*(x{:d}[k-{:d}]-{:.2e})*(x{:d}[k-{:d}]-{:.2e})".format(
                a,argList[0]["input_channel"]+1,argList[0]["delay"],argShift[0],
                argList[1]["input_channel"]+1,argList[1]["delay"],argShift[1])
        
def fcn_poly33_2(x,a,b,c):
    return a*pow(x[0],1)*pow(x[1],1) \
         + b*pow(x[0],2)*pow(x[1],1) \
         + c*pow(x[0],1)*pow(x[1],2)
def txt_poly33_2(argList,argShift,a,b,c):
    if all(shift == 0 for shift in argShift):
        return "{:.2e}*x{:d}[k-{:d}]*x{:d}[k-{:d}] + " \
                "{:.2e}*x{:d}[k-{:d}]^2*x{:d}[k-{:d}] + " \
                "{:.2e}*x{:d}[k-{:d}]*x{:d}[k-{:d}]^2".format(
                a,argList[0]["input_channel"]+1,argList[0]["delay"],
                argList[1]["input_channel"]+1,argList[1]["delay"],
                b,argList[0]["input_channel"]+1,argList[0]["delay"],
                argList[1]["input_channel"]+1,argList[1]["delay"],
                c,argList[0]["input_channel"]+1,argList[0]["delay"],
                argList[1]["input_channel"]+1,argList[1]["delay"])
    else:
        return "{:.2e}*(x{:d}[k-{:d}]-{:.2e})*(x{:d}[k-{:d}]-{:.2e}) + " \
                "{:.2e}*(x{:d}[k-{:d}]-{:.2e})^2*(x{:d}[k-{:d}]-{:.2e}) + " \
                "{:.2e}*(x{:d}[k-{:d}]-{:.2e})*(x{:d}[k-{:d}]-{:.2e})^2".format(
                a,argList[0]["input_channel"]+1,argList[0]["delay"],argShift[0],
                argList[1]["input_channel"]+1,argList[1]["delay"],argShift[1],
                b,argList[0]["input_channel"]+1,argList[0]["delay"],argShift[0],
                argList[1]["input_channel"]+1,argList[1]["delay"],argShift[1],
                c,argList[0]["input_channel"]+1,argList[0]["delay"],argShift[0],
                argList[1]["input_channel"]+1,argList[1]["delay"],argShift[1])
        
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
                
def fcn_linear_3(x,a):
    return a*x[0]*x[1]*x[2]
def txt_linear_3(argList,argShift,a):
    if all(shift == 0 for shift in argShift):
        return "{:.2e}*x{:d}[k-{:d}]*x{:d}[k-{:d}]*x{:d}[k-{:d}]".format(
                a,argList[0]["input_channel"]+1,argList[0]["delay"],
                argList[1]["input_channel"]+1,argList[1]["delay"],
                argList[2]["input_channel"]+1,argList[2]["delay"])
    else:
        return "{:.2e}*(x{:d}[k-{:d}]-{:.2e})*(x{:d}[k-{:d}]-{:.2e})*(x{:d}[k-{:d}]-{:.2e})".format(
                a,argList[0]["input_channel"]+1,argList[0]["delay"],argShift[0],
                argList[1]["input_channel"]+1,argList[1]["delay"],argShift[1],
                argList[2]["input_channel"]+1,argList[2]["delay"],argShift[2])
#=============================================================================#
    
# Exponentials.
#=============================================================================#
def fcn_exp_1(x,a,b):
    return a*(np.exp(b*x[0])-1)
def txt_exp_1(argList,argShift,a,b):
    if all(shift == 0 for shift in argShift):
        return "{:.2e}*(e^({:.2e}*x{:d}[k-{:d}])-1)".format(
                a,b,argList[0]["input_channel"]+1,argList[0]["delay"])
    else:
        return "{:.2e}*(e^({:.2e}*(x{:d}[k-{:d}]-{:.2e}))-1)".format(
                a,b,argList[0]["input_channel"]+1,argList[0]["delay"],argShift[0])
        
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
#=============================================================================#

# Sinusoidal functions.
#=============================================================================#
# This function is hard to fit, so forms with less parameters are included.
def fcn_sin_1(x,a,b,c):
    return a*np.sin(b*x[0]+c)-a*np.sin(c)
def txt_sin_1(argList,argShift,a,b,c):
    if all(shift == 0 for shift in argShift):
        return "{:.2e}*sin({:.2e}*x{:d}[k-{:d}]+{:.2e})-{:.2e}".format(
                a,b,argList[0]["input_channel"]+1,argList[0]["delay"],c,(a*np.sin(c)))
    else:
        return "{:.2e}*sin({:.2e}*(x{:d}[k-{:d}]-{:.2e})+{:.2e})-{:.2e}".format(
                a,b,argList[0]["input_channel"]+1,argList[0]["delay"],argShift[0],c,(a*np.sin(c)))

def fcn_tanh_1(x,a,b):
    return a*np.tanh(b*x[0])
def txt_tanh_1(argList,argShift,a,b):
    if all(shift == 0 for shift in argShift):
        return "{:.2e}*tanh({:.2e}*x{:d}[k-{:d}])".format(
                a,b,argList[0]["input_channel"]+1,argList[0]["delay"])
    else:
        return "{:.2e}*tanh({:.2e}*(x{:d}[k-{:d}]-{:.2e}))".format(
                a,b,argList[0]["input_channel"]+1,argList[0]["delay"],argShift[0])

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

def fcn_tanh12_2(x,a,b,c,d,e):
    return a*np.tanh(b*x[0]-c)*np.tanh(d*x[1]-e) - a*np.tanh(-c)*np.tanh(-e)
def txt_tanh12_2(argList,argShift,a,b,c,d,e):
    if all(shift == 0 for shift in argShift):
        return "{:.2e}*tanh({:.2e}*x{:d}[k-{:d}]-{:.2e})*" \
                "tanh({:.2e}*x{:d}[k-{:d}]-{:.2e}) - " \
                "{:.2e}".format(
                        a,b,argList[0]["input_channel"]+1,argList[0]["delay"],c,
                        d,argList[1]["input_channel"]+1,argList[1]["delay"],e,
                        (a*np.tanh(-c)*np.tanh(-e)))
    else:
        return "{:.2e}*tanh({:.2e}*(x{:d}[k-{:d}]-{:.2e})-{:.2e})*" \
                "tanh({:.2e}*(x{:d}[k-{:d}]-{:.2e})-{:.2e}) - " \
                "{:.2e}".format(
                        a,b,argList[0]["input_channel"]+1,argList[0]["delay"],argShift[0],c,
                        d,argList[1]["input_channel"]+1,argList[1]["delay"],argShift[1],e,
                        (a*np.tanh(-c)*np.tanh(-e)))

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
#=============================================================================#

# Return a list containing template functions for argCount number of arguments.
def fitting_functions():
    
    #=========================================================================#
    # Functions of 1 variable.
    #=========================================================================#
    dct_poly1_1 = {
        "txt": "a*x1",
        "txt_fcn": txt_poly_1,
        "fcn": fcn_poly1_1,
        "upper": [10],
        "lower": [-10],
        "weight": 1.0
    }
    dct_poly2_1 = {
        "txt": "a*x1^2 + b*x1",
        "txt_fcn": txt_poly2_1,
        "fcn": fcn_poly2_1,
        "upper": [10, 10],
        "lower": [-10, -10],
        "weight": 0.55
    }
    dct_squared_1 = {
        "txt": "a*x1^2",
        "txt_fcn": txt_squared_1,
        "fcn": fcn_squared_1,
        "upper": [10],
        "lower": [-10],
        "weight": 1.0
    }
    dct_poly3_1 = {
        "txt": "a*x1^3 + b*x1^2 + c*x1",
        "txt_fcn": txt_poly3_1,
        "fcn": fcn_poly3_1,
        "upper": [10, 10, 10],
        "lower": [-10, -10, -10],
        "weight": 0.20
    }
    dct_cubed_1 = {
        "txt": "a*x1^3",
        "txt_fcn": txt_cubed_1,
        "fcn": fcn_cubed_1,
        "upper": [10],
        "lower": [-10],
        "weight": 1.0
    }
    dct_poly4_1 = {
        "txt": "a*x1^4 + b*x1^3 + b*x1^2 + c*x1",
        "txt_fcn": txt_poly4_1,
        "fcn": fcn_poly4_1,
        "upper": [100, 100, 10, 10],
        "lower": [-100, -100, -10, -10],
        "weight": 0.15
    }
    dct_poly5_1 = {
        "txt": "a*x1^5 + b*x1^4 + b*x1^3 + b*x1^2 + c*x1",
        "txt_fcn": txt_poly5_1,
        "fcn": fcn_poly5_1,
        "upper": [100, 100, 10, 10, 10],
        "lower": [-100, -100, -10, -10, -10],
        "weight": 0.10
    }
    #=========================================================================#
    dct_exp_1 = {
        "txt": "a*(e^(b*x1)-1)",
        "txt_fcn": txt_exp_1,
        "fcn": fcn_exp_1,
        "upper": [10, 5],
        "lower": [-10, -5],
        "weight": 0.25
    }
    #=========================================================================#
    dct_sin_1 = {
        "txt": "a*sin(b*x1+c) - a*sin(c)",
        "txt_fcn": txt_sin_1,
        "fcn": fcn_sin_1,
        "upper": [10, 10*3.14159, 3.14159],
        "lower": [0, 0, -3.14159],
        "weight": 0.25
    }
    dct_tanh_1 = {
        "txt": "a*tanh(b*x1)",
        "txt_fcn": txt_tanh_1,
        "fcn": fcn_tanh_1,
        "upper": [1000, 10],
        "lower": [-1000, 0.001],
        "weight": 0.25
    }
    dct_tanhx_1 = {
        "txt": "a*tanh(b*(x1+c)) + d*(x1+c) - (a*tanh(b*c)+d*c)",
        "txt_fcn": txt_tanhx_1,
        "fcn": fcn_tanhx_1,
        "upper": [1000, 100, 100, 100],
        "lower": [-1000, 0.001, -100, -100],
        "weight": 0.25
    }
    #=========================================================================#
    
    #=========================================================================#
    # Functions of 2 variables.
    #=========================================================================#
    dct_poly22_2 = {
        "txt": "a*x1*x2",
        "txt_fcn": txt_poly22_2,
        "fcn": fcn_poly22_2,
        "upper": [10],
        "lower": [-10],
        "weight": 1.0
    }
    dct_poly33_2 = {
        "txt": "a*x1*x2 + b*x1^2*x2 + c*x1*x2^2",
        "txt_fcn": txt_poly33_2,
        "fcn": fcn_poly33_2,
        "upper": [10, 10, 10],
        "lower": [-10, -10, -10],
        "weight": 0.5
    }
    dct_poly44_2 = {
        "txt": "a*x1*x2 + b*x1^2*x2 + c*x1*x2^2 + d*x1^3*x2 + e*x1^2*x2^2 + f*x1*x2^3",
        "txt_fcn": txt_poly44_2,
        "fcn": fcn_poly44_2,
        "upper": [10, 10, 10, 10, 10, 10],
        "lower": [-10, -10, -10, -10, -10, -10],
        "weight": 0.4
    }
    dct_poly55_2 = {
        "txt": "a*x1*x2 + b*x1^2*x2 + c*x1*x2^2 + d*x1^3*x2 + e*x1^2*x2^2 + " \
               "f*x1*x2^3 + g*x1^4*x2 + h*x1^3*x2^2 + i*x1^2*x2^3 + j*x1*x2^4",
        "txt_fcn": txt_poly55_2,
        "fcn": fcn_poly55_2,
        "upper": [10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
        "lower": [-10, -10, -10, -10, -10, -10, -10, -10, -10, -10],
        "weight": 0.3
    }
    #=========================================================================#
    dct_exp_lin12_2 = {
        "txt": "a*x1*(e^(x2)-1)",
        "txt_fcn": txt_exp_lin12_2,
        "fcn": fcn_exp_lin12_2,
        "upper": [10],
        "lower": [-10],
        "weight": 0.5
    }
    dct_exp_lin21_2 = {
        "txt": "a*x2*(e^(x1)-1)",
        "txt_fcn": txt_exp_lin21_2,
        "fcn": fcn_exp_lin21_2,
        "upper": [10],
        "lower": [-10],
        "weight": 0.5
    }
    #=========================================================================#
    dct_tanh12_2 = {
        "txt": "a*tanh(b*x1-c)*tanh(d*x2-e) - a*tanh(-c)*tanh(-e)",
        "txt_fcn": txt_tanh12_2,
        "fcn": fcn_tanh12_2,
        "upper": [1000, 1, 100, 1, 100],
        "lower": [-1000, 0.001, -100, 0.001, -100],
        "weight": 0.5
    }
    dct_tanh21_2 = {
        "txt": "a*tanh(b*x2-c)*tanh(d*x1-e) - a*tanh(-c)*tanh(-e)",
        "txt_fcn": txt_tanh21_2,
        "fcn": fcn_tanh21_2,
        "upper": [1000, 1, 100, 1, 100],
        "lower": [-1000, 0.001, -100, 0.001, -100],
        "weight": 0.5
    }
    #=========================================================================#
    
    #=========================================================================#
    # Functions of 3 variables.
    #=========================================================================#
    dct_linear_3 = {
        "txt": "a*x1*x2*x3",
        "txt_fcn": txt_linear_3,
        "fcn": fcn_linear_3,
        "upper": [10],
        "lower": [-10],
        "weight": 1.0
    }
    #=========================================================================#
    
    input_1_list = [dct_poly1_1,
                    dct_poly2_1,
                    dct_poly3_1,
                    dct_poly4_1,
                    dct_poly5_1,
                    dct_squared_1,
                    dct_cubed_1,
                    dct_exp_1,
                    dct_sin_1,
                    dct_tanh_1,
                    dct_tanhx_1,
                    ]
    
    input_2_list = [dct_poly22_2,
                    dct_poly33_2,
                    dct_poly44_2,
                    dct_poly55_2,
                    dct_exp_lin12_2,
                    dct_exp_lin21_2,
                    dct_tanh12_2,
                    dct_tanh21_2,
                    ]
    
    input_3_list = [dct_linear_3]

    functionDictionary = {
        1: input_1_list,
        2: input_2_list,
        3: input_3_list        
    }
    
    return functionDictionary
