#include <iostream>
#include <vector>
#include "math.h"
#include "neuron.h"

using namespace std;

#ifndef __CONV2D_INCLUDED__
#define __CONV2D_INCLUDED__


///////////////////////////////////////////////////////////////////////////////////////////////////////
//	2-Dimensional Convolutional Filter Neuron Class Definition
//////////////////////////////////////////////////////////////////////////////////////////////////////

/*
These neurons connect to summation neurons (generic post convolutional neurons which do not have weights or biases).
Post convolutional neurons have an integer member that indicates which output index it corresponds to in the previous conv layer.

NOTE TO SELF: 
	You should make a class in between neuron and conv_filter_2D called conv_filter and 
	then make conv_filter_2D, conv_filter_1D, and conv_filter_3D subclasses of that.

	Consider ways to make one conv_filter class that takes the number of dimensions / dimensions of the input and convolves with a single function.
*/


class conv_filter_2D: public neuron
	{

	public:
		int input_width;	
		int filter_width;
		int kernel_stride;
		bool using_padding;
		vector<float> outputs;
		vector<float> upstream_gradients;

		
		conv_filter_2D(vector<float> ws, float b, float lr, float b1, float b2, int in_wid, int k_stride, bool use_pad)
			: neuron(b, lr, b1, b2, NULL, NULL) // Initialize the base class.
			{
			/* 
			 * A conv_filter is initialized with:
			 *	a vector of weights (from which kernel size is determined),
			 *	a bias,
			 *	a learning rate,
			 *	a value for beta_one (ADAM optimization)
			 *	a value for beta_two (ADAM optimization)
			 *	a size for the side of an input image (square),
			 *	a step size for the kernel,
			 * 	and a boolean value to say whether or not there will be padding.
			 */

			weights = ws;
			for (int i = 0; i < int(weights.size()); i++)
				{
				weights_first_moments.push_back(0.0);
				weights_second_moments.push_back(0.0);
				}

			using_padding = use_pad;
			input_width = in_wid;
			kernel_stride = k_stride;

			// uses the kernel stride, input side size, and the 
			// filter to determine how many outputs there should be.
			filter_width = int(sqrt(double(ws.size())));
			int pad_thickness = 0;
			if (use_pad) { pad_thickness = 2; }
			int output_width = (in_wid + pad_thickness - filter_width)/k_stride + 1;
			for (int i = 0; i < output_width*output_width; i++)
				{
				outputs.push_back(0.0);
				upstream_gradients.push_back(0.0);
				}
			}


		// One line helper functions.
		float get_output_at_index(int index) { return outputs[index]; }
		void set_output_at_index(float new_output, int index) { outputs[index] = new_output; } 
		void increment_upstream_gradient_at_index(float value, int index) { upstream_gradients[index] += value; }


		void add_input_neuron(neuron * n) 
			{ 
			n->add_ouput_neuron(this);
			input_neurons.push_back(n);
			}


		void update_weights_with_vector_of_upstream_gradients(vector<float> dldws)
			{
			for (int i = 0; i < int(weights.size()); i++)
				{
				float dwi = dldws[i];;
				weights_first_moments[i] = beta_one * weights_first_moments[i] + (1.0 - beta_one) * dwi;
				weights_second_moments[i] = beta_two * weights_second_moments[i] + (1.0 - beta_two) * dwi * dwi;
				float first_unbias = weights_first_moments[i] / (1.0 - pow(beta_one, num_updates));
				float second_unbias = weights_second_moments[i] / (1.0 - pow(beta_two, num_updates));
				dwi = first_unbias / (sqrt(second_unbias) + 0.0000001);
				weights[i] -= learning_rate * dwi;
				}
			}


		void back_prop()
			{

                        // Increment the number of updates.
                        num_updates += 1.0;
                        
                        // Update the bias using the sum of the upstream gradients.
			float sum_of_grads = 0.0;
			for (int i = 0; i < int(upstream_gradients.size()); i++) { sum_of_grads += upstream_gradients[i]; }
                        update_bias_with_upstream_gradient(sum_of_grads);

                        // Calculate the upstream gradients for each input and weight.
			vector<float> dldxs;	// Keeps track of gradients to pass to inputs.
			for (int i = 0; i < int(input_neurons.size()); i++) { dldxs.push_back(0.0); }
			vector<float> dldws;	// Keeps track of gradients with respect to ws.
			for (int i = 0; i < int(weights.size()); i++) { dldws.push_back(0.0); }
			for (int row = 0; row < (input_width - filter_width + 1); row++)
                		{
                		for (int col = 0; col < (input_width - filter_width + 1); col++)
                        		{
                        		int output_index = col + row*(input_width - filter_width + 1);
                        		for (int filt_row = 0; filt_row < filter_width; filt_row++)
                                		{
                                		for (int filt_col = 0; filt_col < filter_width; filt_col++)
                                        	     {
                                        	     int weight_index = filt_col + filt_row*filter_width;
                                        	     int input_index = col + filt_col + (row + filt_row)*input_width;
						     dldxs[input_index] += weights[weight_index] * upstream_gradients[output_index];
						     float the_input = input_neurons[input_index]->get_output();
						     dldws[weight_index] += the_input * upstream_gradients[output_index]; 
                                        	     }
                                		}
					}
				}

			// Update each weight using ADAM optimization.
			update_weights_with_vector_of_upstream_gradients(dldws);

			// Update the upstream gradients for each input.
			for (int i = 0; i < int(input_neurons.size()); i++) 
				{ 
				input_neurons[i]->increment_upstream_gradient(dldxs[i]); 
				}

			// Reset the upstream gradients to zero.
			for (int i = 0; i < int(upstream_gradients.size()); i++) { upstream_gradients[i] = 0.0; }
			}


		void forward_prop()
			{
			// Reset the outputs to zero.
			for (int i = 0; i < int(outputs.size()); i++) { outputs[i] = 0.0; }

			// Run the filter over the input vector.
			for (int row = 0; row < (input_width - filter_width + 1); row++)
                		{
                		for (int col = 0; col < (input_width - filter_width + 1); col++)
                        		{
                        		int output_index = col + row*(input_width - filter_width + 1);
                        		for (int filt_row = 0; filt_row < filter_width; filt_row++)
                                		{
                                		for (int filt_col = 0; filt_col < filter_width; filt_col++)
                                        	     {
                                        	     int weight_index = filt_col + filt_row*filter_width;
                                        	     int input_index = col + filt_col + (row + filt_row)*input_width;
						     float out_val = weights[weight_index]*input_neurons[input_index]->get_output();
                                        	     outputs[output_index] += out_val;
                                        	     }
                                		}
                        		outputs[output_index] += bias;
                        		}
                		}
			}

	};

#endif
