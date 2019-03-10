#include <iostream>
#include "math.h"


using namespace std;


#ifndef __act_funcs_INCLUDED__
#define __act_funcs_INCLUDED__


#define alpha 0.05


//-----------------------------------------------------------------------------
// sigmoid activation function
//-----------------------------------------------------------------------------
float sigmoid(float z) { return 1.0 / (1.0 + exp(-z)); }
float sigmoid_derivative(float z) { return (1 - sigmoid(z)) * sigmoid(z); }


//-----------------------------------------------------------------------------
// tanh activation function
//-----------------------------------------------------------------------------
float hyperbolic_tan(float z) { return (2.0 / (1 + exp(-2*z))) - 1.0; }
float hyperbolic_tan_derivative(float z) { return 1.0 - hyperbolic_tan(z)*hyperbolic_tan(z); }


//-----------------------------------------------------------------------------
// relu activation function
//-----------------------------------------------------------------------------
float relu(float z) { return ( (z >= 0.0) ? z : 0.0 ); }
float relu_derivative(float z) { return ( (z >= 0.0) ? 1.0 : 0.0 ); }


//-----------------------------------------------------------------------------
// parametric relu activation function
//-----------------------------------------------------------------------------
float prelu(float z) { return ( (z >= 0.0) ? z : alpha*z); }
float prelu_derivative(float z) { return ( (z >= 0.0) ? 1.0 : alpha ); }


//-----------------------------------------------------------------------------
// exponential linear unit  activation function
//-----------------------------------------------------------------------------
float elu(float z) { return ( (z >= 0.0) ? z : (alpha*(exp(z) - 1.0)) ); }
float elu_derivative(float z) { return ( (z >= 0.0) ? 1 : (elu(z) + alpha) ); }


//-----------------------------------------------------------------------------
// softplus activation function
//-----------------------------------------------------------------------------
float softplus(float z) { return log(1.0 + exp(z)); }
float softplus_derivative(float z) { return 1.0 / (1 + exp(-z)); }


#endif

