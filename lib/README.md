# Fitting Functions Library

This document lists all the template fitting functions (`fcn_` functions) available in `fitting_functions.py`. These functions are used for curve fitting in the equation generation framework.

## Polynomial Functions (1 variable)

### fcn_poly1_1(x, a)
Linear function:

$ax_0$

### fcn_poly2_1(x, a, b)
Quadratic polynomial:

$ax_0^2 + bx_0$

### fcn_squared_1(x, a)
Pure squared term:

$ax_0^2$

### fcn_poly3_1(x, a, b, c)
Cubic polynomial:

$ax_0^3 + bx_0^2 + cx_0$

### fcn_cubed_1(x, a)
Pure cubic term:

$ax_0^3$

### fcn_poly4_1(x, a, b, c, d)
Quartic polynomial:

$ax_0^4 + bx_0^3 + cx_0^2 + dx_0$

### fcn_poly5_1(x, a, b, c, d, e)
Quintic polynomial:

$ax_0^5 + bx_0^4 + cx_0^3 + dx_0^2 + ex_0$

## Polynomial Functions (2 variables)

### fcn_poly22_2(x, a)
Bilinear term:

$ax_0x_1$

### fcn_poly33_2(x, a, b, c)
Second-order bivariate polynomial:

$ax_0x_1 + bx_0^2x_1 + cx_0x_1^2$

### fcn_poly44_2(x, a, b, c, d, e, f)
Third-order bivariate polynomial:

$ax_0x_1 + bx_0^2x_1 + cx_0x_1^2 + dx_0^3x_1 + ex_0^2x_1^2 + fx_0x_1^3$

### fcn_poly55_2(x, a, b, c, d, e, f, g, h, i, j)
Fourth-order bivariate polynomial:

$ax_0x_1 + bx_0^2x_1 + cx_0x_1^2 + dx_0^3x_1 + ex_0^2x_1^2 + fx_0x_1^3 + gx_0^4x_1 + hx_0^3x_1^2 + ix_0^2x_1^3 + jx_0x_1^4$

## Multilinear Functions

### fcn_linear_3(x, a)
Three-variable product:

$ax_0x_1x_2$

### fcn_linear_4(x, a)
Four-variable product:

$ax_0x_1x_2x_3$

### fcn_linear_5(x, a)
Five-variable product:

$ax_0x_1x_2x_3x_4$

## Exponential Functions

### fcn_exp_1(x, a, b)
Exponential function:

$a(e^{bx_0} - 1)$

### fcn_exp_lin12_2(x, a)
Linear-exponential product:

$ax_0(e^{x_1} - 1)$

### fcn_exp_lin21_2(x, a)
Linear-exponential product (reversed):

$ax_1(e^{x_0} - 1)$

## Trigonometric and Hyperbolic Functions

### fcn_sin_1(x, a, b, c)
Sinusoidal function:

$a\sin(bx_0 + c) - a\sin(c)$

### fcn_tanh_1(x, a, b)
Hyperbolic tangent:

$a\tanh(bx_0)$

### fcn_tanhx_1(x, a, b, c, d)
Extended hyperbolic tangent with linear term:

$a\tanh(b(x_0 + c)) + d(x_0 + c) - (a\tanh(bc) + dc)$

### fcn_tanh12_2(x, a, b, c, d, e)
Product of two hyperbolic tangents:

$a\tanh(bx_0 - c)\tanh(dx_1 - e) - a\tanh(-c)\tanh(-e)$

### fcn_tanh21_2(x, a, b, c, d, e)
Product of two hyperbolic tangents (reversed order):

$a\tanh(bx_1 - c)\tanh(dx_0 - e) - a\tanh(-c)\tanh(-e)$

## Notes

- Each function has a corresponding `txt_` function that generates a text representation of the fitted equation.
- The naming convention indicates:
  - The function type (poly, exp, sin, tanh, etc.)
  - The number of input variables (1-5) as a suffix
