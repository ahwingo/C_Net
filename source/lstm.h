#include <iostream>
#include <vector>
#include "math.h"
#include "activation_functions.h"
#include "neuron.h"

using namespace std;


#ifndef __LSTM_LAYER_INCLUDED__
#define __LSTM_LAYER_INCLUDED__

///////////////////////////////////////////////////////////////////////////////////////////////////////
//	LSTM Layer Class Definition
///////////////////////////////////////////////////////////////////////////////////////////////////////

class lstm_layer
	{

	public:

		int time_depth;								// len of seq recurred over, TD

		vector<neuron *> input_neurons;			// R ^ N
		vector<float> vector_of_inputs_at_time;		// R ^ ((N+M)*TD)	// TD vectors of R ^ (N+M)

		vector<neuron *> output_neurons;		// R ^ M	
		vector<float> outputs;				// R ^ M		// uses to hold the output at T - 1 for use on the T timestep. initialized to 0.0


		vector<float> cell_states;			// R ^ (M*(TD+1))	// TD + 1 vectors of R ^ M  (+ 1 for c_t = 0)
		vector<float> cell_states_dldc;			// R ^ M		// stores the error at the cell states when for timestep t = T + 1 

		vector<float> upstream_gradients_recurrent;	// R ^ M		// vector for upstream gradients from the t = T + 1 timestep / for the t = T - 1 timestep

		vector<float> weights_i_gate;			// R ^ ((N+M)*M)	// i gate weight matrix
		vector<float> weights_i_first_moments;		// R ^ ((N+M)*M)	// param for ADAM optimization
		vector<float> weights_i_second_moments;		// R ^ ((N+M)*M)	// param for ADAM optimization
		vector<float> weight_i_dldws;			// R ^ ((N+M)*M)	// accumulates d_loss/d_wi for all td timesteps

		vector<float> weights_f_gate;			// R ^ ((N+M)*M)	// f gate weight matrix
		vector<float> weights_f_first_moments;		// R ^ ((N+M)*M)	// param for ADAM optimization
		vector<float> weights_f_second_moments;		// R ^ ((N+M)*M)	// param for ADAM optimization
		vector<float> weight_f_dldws;			// R ^ ((N+M)*M)	// accumulates d_loss/d_wi for all td timesteps

		vector<float> weights_g_gate;			// R ^ ((N+M)*M)	// g gate weight matrix
		vector<float> weights_g_first_moments;		// R ^ ((N+M)*M)	// param for ADAM optimization
		vector<float> weights_g_second_moments;		// R ^ ((N+M)*M)	// param for ADAM optimization
		vector<float> weight_g_dldws;			// R ^ ((N+M)*M)	// accumulates d_loss/d_wi for all td timesteps

		vector<float> weights_o_gate;			// R ^ ((N+M)*M)	// o gate weight matrix
		vector<float> weights_o_first_moments;		// R ^ ((N+M)*M)	// param for ADAM optimization
		vector<float> weights_o_second_moments;		// R ^ ((N+M)*M)	// param for ADAM optimization
		vector<float> weight_o_dldws;			// R ^ ((N+M)*M)	// accumulates d_loss/d_wi for all td timesteps
	
	
		vector<float> biases_i;				// R ^ M		// vector of i gate biases
		vector<float> biases_i_first_moments;		// R ^ M		// param for ADAM optimization
		vector<float> biases_i_second_moments;		// R ^ M		// param for ADAM optimization
		vector<float> bias_i_dldbs;			// R ^ M		// accumulates d_loss/d_wi for all td timesteps

		vector<float> biases_f;				// R ^ M		// vector of f gate biases
		vector<float> biases_f_first_moments;		// R ^ M		// param for ADAM optimization	
		vector<float> biases_f_second_moments;		// R ^ M		// param for ADAM optimization
		vector<float> bias_f_dldbs;			// R ^ M		// accumulates d_loss/d_wi for all td timesteps

		vector<float> biases_g;				// R ^ M		// vector of g gate biases
		vector<float> biases_g_first_moments;		// R ^ M		// param for ADAM optimization
		vector<float> biases_g_second_moments;		// R ^ M		// param for ADAM optimization
		vector<float> bias_g_dldbs;			// R ^ M		// accumulates d_loss/d_wi for all td timesteps

		vector<float> biases_o;				// R ^ M		// vector of o gate biases
		vector<float> biases_o_first_moments;		// R ^ M		// param for ADAM optimization
		vector<float> biases_o_second_moments;		// R ^ M		// param for ADAM optimization
		vector<float> bias_o_dldbs;			// R ^ M		// accumulates d_loss/d_wi for all td timesteps


		vector<float> pre_i_gate_values;		// R ^ (M*TD)		// intermediate values cached for backpropagation through time
		vector<float> pre_f_gate_values;		// R ^ (M*TD)           // intermediate values cached for backpropagation through time
		vector<float> pre_g_gate_values;		// R ^ (M*TD)		// intermediate values cached for backpropagation through time
		vector<float> pre_o_gate_values;		// R ^ (M*TD)		// intermediate values cached for backpropagation through time


		float learning_rate;							// neuron's learning rate
		
		float beta_one;								// param for ADAM optimization
		float beta_two;								// param for ADAM optimization	
		float num_updates;							// param for ADAM optimization


		lstm_layer(int td, int num_outputs, int num_inputs, float lr, float b1, float b2, normal_distribution<float> dist, default_random_engine * generator)
			{

                        /* 
                         * An lstm_layer is initialized with:
			 * 	a time depth,
			 *	a size for the number of outputs
			 *	a size for the number of inputs
                         *      a learning rate,
                         *      a value for beta_one (ADAM optimization)
                         *      a value for beta_two (ADAM optimization)
			 *	and a distribution for param initialization.
                         */

			// Initialize the basic values.
			time_depth = td;
			learning_rate = lr;
			beta_one = b1;
			beta_two = b2;
			num_updates = 0;

			// Initialize the outputs and cell_states with 0.0's for the t = 0 time step.
			for (int i = 0; i < num_outputs; i++)
				{
				cell_states.push_back(0.0);
				}

			// Initialize the weight matrices, their moments and their error accumulators.
			int weights_len = num_inputs + num_outputs;
			int weights_wid = num_outputs; 
			for (int i = 0; i < weights_len*weights_wid; i++)
				{
				// i gate
				weights_i_gate.push_back(dist(*generator));		
				weights_i_first_moments.push_back(0.0);	
				weights_i_second_moments.push_back(0.0);	
				weight_i_dldws.push_back(0.0);	
	
				// f gate
				weights_f_gate.push_back(dist(*generator));		
				weights_f_first_moments.push_back(0.0);	
				weights_f_second_moments.push_back(0.0);	
				weight_f_dldws.push_back(0.0);	

				// g gate
				weights_g_gate.push_back(dist(*generator));		
				weights_g_first_moments.push_back(0.0);	
				weights_g_second_moments.push_back(0.0);	
				weight_g_dldws.push_back(0.0);	

				// o gate
				weights_o_gate.push_back(dist(*generator));		
				weights_o_first_moments.push_back(0.0);	
				weights_o_second_moments.push_back(0.0);	
				weight_o_dldws.push_back(0.0);	
				}
	
			// Initialize the bias vectors and their moments and their error accumulators.
			for (int i = 0; i < num_outputs; i++)
				{
				// i gate
				biases_i.push_back(dist(*generator));
				biases_i_first_moments.push_back(0.0);
				biases_i_second_moments.push_back(0.0);
				bias_i_dldbs.push_back(0.0);

				// f gate
				biases_f.push_back(dist(*generator));
				biases_f_first_moments.push_back(0.0);
				biases_f_second_moments.push_back(0.0);
				bias_f_dldbs.push_back(0.0);

				// g gate
				biases_g.push_back(dist(*generator));
				biases_g_first_moments.push_back(0.0);
				biases_g_second_moments.push_back(0.0);
				bias_g_dldbs.push_back(0.0);

				// o gate
				biases_o.push_back(dist(*generator));
				biases_o_first_moments.push_back(0.0);
				biases_o_second_moments.push_back(0.0);
				bias_o_dldbs.push_back(0.0);
				}

			// Initialize a vector of output neurons. 
			// Pass this layer of output neurons to the network that initializes it. 
			// Set its output like you would a set of inputs to the network.
			// Also initialize the vector of recurrent upstream gradients.
			// Also initialize the cell states error vector.
			for (int i = 0; i < num_outputs; i++)
				{
				neuron * an_output_neuron = new neuron(0.0, 0.0, 0.0, 0.0, NULL, NULL);
				output_neurons.push_back(an_output_neuron);	
			
				outputs.push_back(0.0);

				upstream_gradients_recurrent.push_back(0.0);
				cell_states_dldc.push_back(0.0);
				}
		} // end of lstm_layer initializer


	
		vector<neuron *> get_layer_of_output_neurons() { return output_neurons; }

		void add_input_neuron(neuron * n) { input_neurons.push_back(n); }


		// These two functions are needed to make copies of the lstm.
		void replace_input_neuron_at_index(neuron * n, int index) { input_neurons[index] = n; }
		void replace_output_neuron_at_index(neuron * n, int index) { output_neurons[index] = n; }	


		vector<float> calc_i_vector(vector<float> full_input)
			{
			vector<float> i_vals = biases_i;
			for(int row = 0; row < int(i_vals.size()); row++)
				{
				for (int col = 0; col < int(full_input.size()); col++)
					{
					i_vals[row] += full_input[col] * weights_i_gate[row*int(full_input.size()) + col];
					}
				}
			// cache these pre non-linear function values for use later during back prop
			for (int i = 0; i < int(i_vals.size()); i++) { pre_i_gate_values.push_back(i_vals[i]); }
			// apply the sigmoid non linearity to the i_vals vector
			for (int i = 0; i < int(i_vals.size()); i++) { i_vals[i] = sigmoid(i_vals[i]); }
			// return the activated vector
			return i_vals;
			}


		vector<float> calc_f_vector(vector<float> full_input)
			{
			vector<float> f_vals = biases_f;
			for(int row = 0; row < int(f_vals.size()); row++)
				{
				for (int col = 0; col < int(full_input.size()); col++)
					{
					f_vals[row] += full_input[col] * weights_f_gate[row*int(full_input.size()) + col];
					}
				}
			// cache these pre non-linear function values for use later during back prop
			for (int i = 0; i < int(f_vals.size()); i++) { pre_f_gate_values.push_back(f_vals[i]); }
			// apply the sigmoid non linearity to the i_vals vector
			for (int i = 0; i < int(f_vals.size()); i++) { f_vals[i] = sigmoid(f_vals[i]); }
			// return the activated vector
			return f_vals;
			}


		vector<float> calc_g_vector(vector<float> full_input)
			{
			vector<float> g_vals = biases_g;
			for(int row = 0; row < int(g_vals.size()); row++)
				{
				for (int col = 0; col < int(full_input.size()); col++)
					{
					g_vals[row] += full_input[col] * weights_g_gate[row*int(full_input.size()) + col];
					}
				}
			// cache these pre non-linear function values for use later during back prop
			for (int i = 0; i < int(g_vals.size()); i++) { pre_g_gate_values.push_back(g_vals[i]); }
			// apply the sigmoid non linearity to the i_vals vector
			for (int i = 0; i < int(g_vals.size()); i++) { g_vals[i] = hyperbolic_tan(g_vals[i]); }
			// return the activated vector
			return g_vals;
			}


		vector<float> calc_o_vector(vector<float> full_input)
			{
			vector<float> o_vals = biases_o;
			for(int row = 0; row < int(o_vals.size()); row++)
				{
				for (int col = 0; col < int(full_input.size()); col++)
					{
					o_vals[row] += full_input[col] * weights_o_gate[row*int(full_input.size()) + col];
					}
				}
			// cache these pre non-linear function values for use later during back prop
			for (int i = 0; i < int(o_vals.size()); i++) { pre_o_gate_values.push_back(o_vals[i]); }
			// apply the sigmoid non linearity to the i_vals vector
			for (int i = 0; i < int(o_vals.size()); i++) { o_vals[i] = sigmoid(o_vals[i]); }
			// return the activated vector
			return o_vals;
			}
		
	
		vector<float> calc_cell_state(vector<float> i_vals, vector<float> f_vals, vector<float> g_vals)
			{
			vector<float> c_t;
			int output_size = int(i_vals.size());
			int start_index = int(cell_states.size()) - output_size;
			for (int i = 0; i < output_size; i++)
				{
				float i_times_g = i_vals[i] * g_vals[i];
				float f_times_prev_c = f_vals[i] * cell_states[start_index + i];
				c_t.push_back(i_times_g + f_times_prev_c);
				}

			// Cache the new cell state.
			for (int i = 0; i < output_size; i++) { cell_states.push_back(c_t[i]); }

			// Return the new cell state.
			return c_t;
			}


		void calc_output(vector<float> c_t, vector<float> o_vals)
			{
			for (int i = 0; i < int(c_t.size()); i++)
				{
				// Set the output values of our stored output layer neurons, so the other layers can access it.
				output_neurons[i]->set_output(hyperbolic_tan(c_t[i]) * o_vals[i]);
				outputs[i] = hyperbolic_tan(c_t[i]) * o_vals[i];
				}
			}


		void forward_prop()
			{
			// Add inputs from T and outputs from T - 1 to the list of all inputs so we can do back prop through time.
			for (int i = 0; i < int(input_neurons.size()); i++) { vector_of_inputs_at_time.push_back(input_neurons[i]->get_output()); }
			for (int i = 0; i < int(output_neurons.size()); i++) { vector_of_inputs_at_time.push_back(outputs[i]); }	

			// Add inputs from T and outputs from T - 1 to the following vector so that its easy to use the functions below.
			vector<float> full_input_vector;
			for (int i = 0; i < int(input_neurons.size()); i++) { full_input_vector.push_back(input_neurons[i]->get_output()); }
			for (int i = 0; i < int(output_neurons.size()); i++) { full_input_vector.push_back(outputs[i]); }

			// Calculate i, f, g, and o gate vectors and cache intermediate values.
			vector<float> i_gate_output = calc_i_vector(full_input_vector);
			vector<float> f_gate_output = calc_f_vector(full_input_vector);
			vector<float> g_gate_output = calc_g_vector(full_input_vector);
			vector<float> o_gate_output = calc_o_vector(full_input_vector);
	
			// Calculate the new cell state using the i, f, g, and c_t_min_1 vector and cache values.
			vector<float> c_t = calc_cell_state(i_gate_output, f_gate_output, g_gate_output);
		
			// Calculate the new output vector and cache values.
			calc_output(c_t, o_gate_output);
			}


		// This function pops items off the cell states vector.
		vector<float> get_c_t_vector()
			{
			vector<float> ret_val;
			int start_index = int(cell_states.size()) - int(output_neurons.size());
			int end_index = int(cell_states.size());

			// Populate the return vector in the correct order.
			for (int i = start_index; i < end_index; i++)
				{
				ret_val.push_back(cell_states[i]);
				}

			// Pop old values off the stack.
			for (int i = 0; i < int(output_neurons.size()); i++)
				{
				cell_states.pop_back();
				}

			return ret_val;
			}
			
	
		// This function returns a vector of c_t_minus_one without removing them from the stack.
		vector<float> get_c_t_minus_one_vector()
			{
			vector<float> ret_val;
			int start_index = int(cell_states.size()) - int(output_neurons.size());
			int end_index = int(cell_states.size());

			// Populate the return vector in the correct order.
			for (int i = start_index; i < end_index; i++)
				{
				ret_val.push_back(cell_states[i]);
				}

			return ret_val;
			}


		// This function gets pre i gate values for timestep t = T and pops them off the stack.
		vector<float> get_pre_i_gate_t_vector()
			{
			vector<float> ret_val;
			int start_index = int(pre_i_gate_values.size()) - int(output_neurons.size());
			int end_index = int(pre_i_gate_values.size());

			// Populate the return vector in the correct order.
			for (int i = start_index; i < end_index; i++)
				{
				ret_val.push_back(pre_i_gate_values[i]);
				}

			// Pop old values off the stack.
			for (int i = 0; i < int(output_neurons.size()); i++)
				{
				pre_i_gate_values.pop_back();
				}

			return ret_val;
			}


		// This function returns the activation of the i gate given pre_i_gate vector.
		vector<float> get_post_i_gate_t_vector(vector<float> pre_i_gate_t)
			{
			vector<float> ret_val;
			for (int i = 0; i < int(pre_i_gate_t.size()); i++) { ret_val.push_back(sigmoid(pre_i_gate_t[i])); }
			return ret_val;
			}


		// This function gets pre g gate values for timestep t = T and pops them off the stack.
		vector<float> get_pre_g_gate_t_vector()
			{
			vector<float> ret_val;
			int start_index = int(pre_g_gate_values.size()) - int(output_neurons.size());
			int end_index = int(pre_g_gate_values.size());

			// Populate the return vector in the correct order.
			for (int i = start_index; i < end_index; i++)
				{
				ret_val.push_back(pre_g_gate_values[i]);
				}

			// Pop old values off the stack.
			for (int i = 0; i < int(output_neurons.size()); i++)
				{
				pre_g_gate_values.pop_back();
				}

			return ret_val;
			}


		// This function returns the activation of the g gate given pre_i_gate vector.
		vector<float> get_post_g_gate_t_vector(vector<float> pre_g_gate_t)
			{
			vector<float> ret_val;
			for (int i = 0; i < int(pre_g_gate_t.size()); i++) { ret_val.push_back(hyperbolic_tan(pre_g_gate_t[i])); }
			return ret_val;
			}

		// this function gets pre o gate values for timestep t = T and pops them off the stack
		vector<float> get_pre_o_gate_t_vector()
			{
			vector<float> ret_val;
			int start_index = int(pre_o_gate_values.size()) - int(output_neurons.size());
			int end_index = int(pre_o_gate_values.size());

			// populate the return vector in the correct order
			for (int i = start_index; i < end_index; i++)
				{
				ret_val.push_back(pre_o_gate_values[i]);
				}

			// pop old values off the stack
			for (int i = 0; i < int(output_neurons.size()); i++)
				{
				pre_o_gate_values.pop_back();
				}

			return ret_val;
			}


		// This function returns the activation of the o gate given pre_o_gate vector				
		vector<float> get_post_o_gate_t_vector(vector<float> pre_o_gate_t)
			{
			vector<float> ret_val;
			for (int i = 0; i < int(pre_o_gate_t.size()); i++) { ret_val.push_back(sigmoid(pre_o_gate_t[i])); }
			return ret_val;
			}
		
	
		// This function gets post f gate values (from the pre gate stack) for timestep t = T+1 and pops them off the pre gate stack
		vector<float> get_post_f_gate_t_plus_one_vector()
			{
			vector<float> ret_val;

			// First, check to see that we are not calling backprop for the first time on this sequence. Otherwise, F will always be larger by a whole timestep.
			if (int(pre_f_gate_values.size()) == int(pre_i_gate_values.size()))			
				{
				for (int i = 0; i < int(output_neurons.size()); i++) { ret_val.push_back(0.0); }				
				return ret_val;
				}

			// otherwise, it is okay to pop the current elements off the stack
			int start_index = int(pre_f_gate_values.size()) - int(output_neurons.size());
			int end_index = int(pre_f_gate_values.size());

			// populate the return vector in the correct order
			for (int i = start_index; i < end_index; i++)
				{
				ret_val.push_back(sigmoid(pre_f_gate_values[i]));
				}

			// pop old values off the stack
			for (int i = 0; i < int(output_neurons.size()); i++)
				{
				pre_f_gate_values.pop_back();
				}

			return ret_val;
			}


		// gets the pre_f_gate values from this timestep and returns them.
		vector<float> get_pre_f_gate_t_vector()
			{	
			vector<float> ret_val;
			int start_index = int(pre_f_gate_values.size()) - int(output_neurons.size());
			int end_index = int(pre_f_gate_values.size());

			// populate the return vector in the correct order
			for (int i = start_index; i < end_index; i++)
				{
				ret_val.push_back(pre_f_gate_values[i]);
				}

			return ret_val;
			}


		// Gets the inputs from this timestep (both actual and recurrent) and pops them off the stack
		vector<float> get_inputs_from_t()
			{
			vector<float> ret_val;
			int start_index = int(vector_of_inputs_at_time.size()) - int(output_neurons.size()) - int(input_neurons.size());
			int end_index = int(vector_of_inputs_at_time.size());
		
			// Populate the return vector in the correct order
			for (int i = start_index; i < end_index; i++)
				{
				ret_val.push_back(vector_of_inputs_at_time[i]);
				}	
			
			// Pop old values off the stack
			for (int i = 0; i < (int(output_neurons.size()) + int(input_neurons.size())); i++)
				{
				vector_of_inputs_at_time.pop_back();
				} 

			return ret_val;
			}


		// Returns the total error at this timestep.
		vector<float> get_total_error_for_layer()
			{
			// the total error is the sum of the error at the output neurons and the values in the upstream_gradients_recurrent vector
			vector<float> ret_val;
			for (int i = 0; i < int(output_neurons.size()); i++)
				{
				float error = output_neurons[i]->get_upstream_gradient() + upstream_gradients_recurrent[i];
				ret_val.push_back(error);
				}
			return ret_val;
			}


		// Find the pre o gate error.
		vector<float> calc_error_o_gate(vector<float> total_error_for_layer, vector<float> c_t, vector<float> pre_o_gate_t)
			{
			vector<float> ret_val;
			for (int i = 0; i < int(output_neurons.size()); i++)
				{
				float error = total_error_for_layer[i] * hyperbolic_tan(c_t[i]) * sigmoid_derivative(pre_o_gate_t[i]);
				ret_val.push_back(error);
				}
			return ret_val;			
			}


		// Find the error at the cell state.
		vector<float> calc_cell_states_error_at_t(vector<float> tot_err_for_layer, vector<float> post_o, vector<float> c_t, vector<float> dldc, vector<float> f_plus_one)
			{
			vector<float> ret_val;
			for (int i = 0; i < int(output_neurons.size()); i++)
				{
				float error = tot_err_for_layer[i] * post_o[i] * hyperbolic_tan_derivative(c_t[i]);
				error += dldc[i] * f_plus_one[i];
				ret_val.push_back(error);
				}
			return ret_val;
			}


		// Find pre f gate error.
		vector<float> calc_error_f_gate(vector<float> dldc_t, vector<float> c_t_minus_one, vector<float> pre_f_gate_t)
			{
			vector<float> ret_val;
			for (int i = 0; i < int(output_neurons.size()); i++)
				{
				float error = dldc_t[i] * c_t_minus_one[i] * sigmoid_derivative(pre_f_gate_t[i]);
				ret_val.push_back(error);	
				}
			return ret_val;
			}


		// Find pre i gate error.
		vector<float> calc_error_i_gate(vector<float> dldc_t, vector<float> post_g_gate_t, vector<float> pre_i_gate_t)
			{
			vector<float> ret_val;
			for (int i = 0; i < int(output_neurons.size()); i++)
				{
				float error = dldc_t[i] * post_g_gate_t[i] * sigmoid_derivative(pre_i_gate_t[i]);
				ret_val.push_back(error);	
				}
			return ret_val;
			}

		
		// Find pre g gate error. notice how this function does the same thing as some of these other functions... Room to simplify.
		vector<float> calc_error_g_gate(vector<float> dldc_t, vector<float> post_i_gate_t, vector<float> pre_g_gate_t)
			{
			vector<float> ret_val;
			for (int i = 0; i < int(output_neurons.size()); i++)
				{
				float error = dldc_t[i] * post_i_gate_t[i] * hyperbolic_tan_derivative(pre_g_gate_t[i]);
				ret_val.push_back(error);	
				}
			return ret_val;
			}


		// This helper function calculates the outer product of two matrices.
		vector<float> outer_product(vector<float> xs, vector<float> ys)
			{
			vector<float> ret_val;
			for (int x = 0; x < int(xs.size()); x++)
				{
				for (int y = 0; y < int(ys.size()); y++)
				{
				ret_val.push_back(xs[x]*ys[y]);
				}
				}
			return ret_val;
			}


		// Increment the errors at the weights. 
		void increment_weights_error_using(vector<float> xs, vector<float> i_err, vector<float> f_err, vector<float> g_err, vector<float> o_err)
			{
			// update the i,f, g, and o gate weight errors
			vector<float> i_updates = outer_product(i_err, xs);
			vector<float> f_updates = outer_product(f_err, xs);
			vector<float> g_updates = outer_product(g_err, xs);
			vector<float> o_updates = outer_product(o_err, xs);
			for (int i = 0; i < int(weight_i_dldws.size()); i++) 
				{ 
				weight_i_dldws[i] += i_updates[i]; 
				weight_f_dldws[i] += f_updates[i];
				weight_g_dldws[i] += g_updates[i];
				weight_o_dldws[i] += o_updates[i];
				}
			return;	
			}


		// Increment the errors at the biases.
		void increment_biases_error_using(vector<float> i_gate_error, vector<float> f_gate_error, vector<float> g_gate_error, vector<float> o_gate_error)
			{
			for (int i = 0; i < int(output_neurons.size()); i++)
				{
				bias_i_dldbs[i] += i_gate_error[i];
				bias_f_dldbs[i] += f_gate_error[i];
				bias_g_dldbs[i] += g_gate_error[i];
				bias_o_dldbs[i] += o_gate_error[i];
				}
			return;
			}


		// This helper function returns the transpose of a matrix. wid = number of columns.  hgt = number of rows.
		vector<float> transpose_matrix_with_orig_wid_and_hgt(vector<float> matrix, int orig_w, int orig_h)
			{
			vector<float> ret_val;
			for (int col = 0; col < orig_w; col++)
				{
				for (int row = 0; row < orig_h; row++)
				{
				int index_to_add = row*orig_w + col;
				ret_val.push_back(matrix[index_to_add]);
				}				
				}
			return ret_val;
			}
		

		// This helper function multiplies two matrices where y1 must equal x2.
		vector<float> multiply_matrices_with_dimensions(vector<float> matrix_1, int x1, int y1, vector<float> matrix_2, int x2, int y2) 
			{
			vector<float> ret_val;
			for (int left_mat_row = 0; left_mat_row < x1; left_mat_row++)
				{
				for (int right_mat_col = 0; right_mat_col < y2; right_mat_col++)
				{
				float val_to_add = 0.0;
				for (int middle_dim = 0; middle_dim < y1; middle_dim ++)
					{
					val_to_add += matrix_1[left_mat_row*y1 + middle_dim] * matrix_2[middle_dim*y2 + right_mat_col];
					}
				ret_val.push_back(val_to_add);
				}
				}
			return ret_val;
			}

		
		// Calculates updates for the input neurons.
		void update_the_upstream_gradients_using(vector<float> i_gate_error, vector<float> f_gate_error, vector<float> g_gate_error, vector<float> o_gate_error)
			{
			// Determine the dimensions of the weight matrices. width = number of columns. height = number of rows.
			int orig_width = int(input_neurons.size()) + int(output_neurons.size());
			int orig_height = int(output_neurons.size());
	
			// Use these as the dimensions when using the transposed vectors.
			int tran_width = orig_height;
			int tran_height = orig_width;

			// Handle the i gate stuff.
			vector<float> i_transposed_weights = transpose_matrix_with_orig_wid_and_hgt(weights_i_gate, orig_width, orig_height);
			vector<float> i_changes = multiply_matrices_with_dimensions(i_transposed_weights, tran_height, tran_width, i_gate_error, tran_width, 1);

			// Handle the f gate stuff.
			vector<float> f_transposed_weights = transpose_matrix_with_orig_wid_and_hgt(weights_f_gate, orig_width, orig_height);
			vector<float> f_changes = multiply_matrices_with_dimensions(f_transposed_weights, tran_height, tran_width, f_gate_error, tran_width, 1);

			// Handle the g gate stuff.
			vector<float> g_transposed_weights = transpose_matrix_with_orig_wid_and_hgt(weights_g_gate, orig_width, orig_height);
			vector<float> g_changes = multiply_matrices_with_dimensions(g_transposed_weights, tran_height, tran_width, g_gate_error, tran_width, 1);

			// Handle the o gate stuff.
			vector<float> o_transposed_weights = transpose_matrix_with_orig_wid_and_hgt(weights_o_gate, orig_width, orig_height);
			vector<float> o_changes = multiply_matrices_with_dimensions(o_transposed_weights, tran_height, tran_width, o_gate_error, tran_width, 1);

			// Pass the gradient back to the input neurons.
			for (int i = 0; i < int(input_neurons.size()); i++)
				{
				float total_update = i_changes[i] + f_changes[i] + g_changes[i] + o_changes[i];
				input_neurons[i]->increment_upstream_gradient(total_update);
				}
	
			// Pass the gradient back in recurrent form.
			for (int i = int(input_neurons.size()); i < int(input_neurons.size()) + int(output_neurons.size()); i++)
				{
				float total_update = i_changes[i] + f_changes[i] + g_changes[i] + o_changes[i];
				upstream_gradients_recurrent[i - int(input_neurons.size())] = total_update;
				}

			return;
			}





		void back_prop()
			{
			// note that we do backprop() for as many times as we have done the forward prop (ie total length of sequence observed).
			// each time we do a forward prop, we append all cached values to their respective vectors.
			// when we do backprop, we do it in reverse order as we did the inputs. so, we can think about it like a stack.
			// thinking of the backprop function as popping items off of a stack, we can call backprop anytime the stack is non empty.
			// because we call backprop on a layer one time step at a time, we are able to call backprop on all the layers in the network,
			// front to back, as long as they have somehow retained their input values at that timestep. 

			// perhaps all neurons (the neuron superclass) could have their input values inserted into a stack, like a short term memory needed for backprop.
			// consider an input image that is 250*250*3 floats (4 bytes). 
			// if we store 100 input instances into the stack, it would take 71.5 MB. That is not too much.

			// also, when do you actually update the parameters? if we should only update the weights at the very end, 
			// then we should store the dw's and db's and be able to call backprop() without updating these parameters.
			// instead, backprop() should just increment the updates for each param, and another function update() should actually update the params.
			// this could be implemented in the neuron superclass as well.
			// because we wait until we roll the network back up before we make param updates, we need to store the dws and dbs, 
			// which accumulate during each backprop. the amount each step adds to the gradient is the full value at that timestep divided by time_depth.
			// use ADAM optimization during the update function call.

			// from: "a gentle introduction to backpropagation through time"
			// TBPTT(n,n): Updates are performed at the end of the sequence across all timesteps in the sequence (e.g. classical BPTT).
			// TBPTT(1,n): timesteps are processed one at a time followed by an update that covers all timesteps seen so far.
			// TBPTT(k1,1): The network likely does not have enough temporal context to learn, relying heavily on internal state and inputs.

			// TBPTT(k1,k2), where k1<k2<n: Multiple updates are performed per sequence which can accelerate training.
			// we may have to find a more clever way to store network values. we dont want to get rid of all elements in the stack (want to leave k2 - k1 timesteps).

			// TBPTT(k1,k2), where k1=k2: A common configuration where a fixed number of timesteps are used for both forward and backward-pass timesteps.
			// this looks like it best fits the above stack styled definition. we will go with this.

			// to make better use of space, we only keep 2 vectors at a time for the upsteam gradients (one vector in R ^ (M*2))
			// let the vector starting at upsteam_gradients[M] and ending on upsteam_gradients[2M-1] be the upstream for the current timestep.
			// when the forward layers increment the upsteam gradient of this layer, they do it through the vector of output_neurons.
			// to get the error at this timestep, we need to  


			// step 0: get all of the vectors that you need and can pop off their stacks. functions pop off stack and reorder as necessary.
			vector<float> c_t = get_c_t_vector();						// this pops items off the stack from: cell_states
			vector<float> c_t_minus_one = get_c_t_minus_one_vector();			// this peeks but does NOT pop items off the stack: cell_states
			vector<float> dldc_t_plus_one = cell_states_dldc;				// values from: cell_states_dldc  TO DO: get rid of this redundancy

			vector<float> post_f_gate_t_plus_one = get_post_f_gate_t_plus_one_vector();	// pops items off the stack and activates from: pre_f_gate_values
			vector<float> pre_f_gate_t = get_pre_f_gate_t_vector();				// peeks but does NOT pop items off the stack: pre_f_gate_values

			vector<float> pre_i_gate_t = get_pre_i_gate_t_vector();				// this pops items off the stack from: pre_i_gate_values
			vector<float> post_i_gate_t = get_post_i_gate_t_vector(pre_i_gate_t);		// this func activates the vector as needed for backprop

			vector<float> pre_g_gate_t = get_pre_g_gate_t_vector();				// pops items off the stack from: pre_g_gate_values
			vector<float> post_g_gate_t = get_post_g_gate_t_vector(pre_g_gate_t); 		// this func activates the vector as needed for backprop

			vector<float> pre_o_gate_t = get_pre_o_gate_t_vector();				// pops items off the stack from: pre_o_gate_values
			vector<float> post_o_gate_t = get_post_o_gate_t_vector(pre_o_gate_t);		// this func activates the vector as needed for backprop
		
			vector<float> inputs_from_this_timestep = get_inputs_from_t();			// this pops the full vect of inputs off of the stack: vector_of_inputs_at_time	

			// step 1: make float vector for the error at the current timestep (δyt = ∆t +RTz δzt+1 +RTi δit+1 +RTf δft+1 +RTo δot+1)
			vector<float> total_error_for_layer = get_total_error_for_layer();

			// step 2: find the error at pre o gate values (δo ̄t = δyt * h(ct) * σ′(o ̄t))
			vector<float> o_gate_error = calc_error_o_gate(total_error_for_layer, c_t, pre_o_gate_t);			

			// step 3: find the error at the cell states,
			vector<float> dldc_t = calc_cell_states_error_at_t(total_error_for_layer, post_o_gate_t, c_t, dldc_t_plus_one, post_f_gate_t_plus_one);

			// step 3.5: reset cell_states_dldc. TO DO: should we do this elementwise instead?
			cell_states_dldc = dldc_t;

			// step 4: find the error at the pre f gate values
			vector<float> f_gate_error = calc_error_f_gate(dldc_t, c_t_minus_one, pre_f_gate_t);
	
			// step 5: find the error at the pre i gate values
			vector<float> i_gate_error = calc_error_i_gate(dldc_t, post_g_gate_t, pre_i_gate_t);			

			// step 6: find the error at the pre g gate values
			vector<float> g_gate_error = calc_error_g_gate(dldc_t, post_i_gate_t, pre_g_gate_t);			

			// step 7: increment the errors at the weights and biases
			increment_weights_error_using(inputs_from_this_timestep, i_gate_error, f_gate_error, g_gate_error, o_gate_error);
			increment_biases_error_using(i_gate_error, f_gate_error, g_gate_error, o_gate_error);

			// step 8: update the upstream gradients of the input neurons and the recurrent upstream gradient
			update_the_upstream_gradients_using(i_gate_error, f_gate_error, g_gate_error, o_gate_error);

			// step 9: set the upstream gradients at the output neurons back to zero?????
			for (int i = 0; i < int(output_neurons.size()); i++) { output_neurons[i]->set_upstream_gradient(0.0); }

			}


		void make_parameter_updates()
			{

			// DEBUGGING
			/*
			cout<<endl<<endl<<"make_parameter_updates()"<<endl<<"all vectors have the following sizes"<<endl;
	
			cout<<"weights_o_gate: "<<weights_o_gate.size()<<endl;
			cout<<"weights_o_first_moments: "<<weights_o_first_moments.size()<<endl;
			cout<<"weights_o_second_moments: "<<weights_o_second_moments.size()<<endl;
			cout<<"weight_o_dldws: "<<weight_o_dldws.size()<<endl;
			cout<<"biases_o: "<<biases_o.size()<<endl;
			cout<<"biases_o_first_moments: "<<biases_o_first_moments.size()<<endl;
			cout<<"biases_o_second_moments: "<<biases_o_second_moments.size()<<endl;
			cout<<"bias_o_dldbs: "<<bias_o_dldbs.size()<<endl;
		
			cout<<endl<<"pre_i_gate_values size: "<<pre_i_gate_values.size()<<endl;
			cout<<endl<<"pre_f_gate_values size: "<<pre_f_gate_values.size()<<endl;
			cout<<endl<<"pre_g_gate_values size: "<<pre_g_gate_values.size()<<endl;
			cout<<endl<<"pre_o_gate_values size: "<<pre_o_gate_values.size()<<endl;

			cout<<"cell_states: "<<cell_states.size()<<endl;
			*/	

			num_updates += 1;
	
			// Update the weights.
			for (int i = 0; i < int(weight_i_dldws.size()); i++)
				{
				// i gate
				weights_i_first_moments[i] = beta_one * weights_i_first_moments[i] + (1.0 - beta_one) * weight_i_dldws[i];
				weights_i_second_moments[i] = beta_two * weights_i_second_moments[i] + (1.0 - beta_two) * weight_i_dldws[i] * weight_i_dldws[i];
				float first_unbias = weights_i_first_moments[i] / (1.0 - pow(beta_one, num_updates));
				float second_unbias = weights_i_second_moments[i] / (1.0 - pow(beta_two, num_updates));
				weights_i_gate[i] -= learning_rate * (first_unbias / (sqrt(second_unbias) + 0.0000001));

				// f gate
				weights_f_first_moments[i] = beta_one * weights_f_first_moments[i] + (1.0 - beta_one) * weight_f_dldws[i];
				weights_f_second_moments[i] = beta_two * weights_f_second_moments[i] + (1.0 - beta_two) * weight_f_dldws[i] * weight_f_dldws[i];
				first_unbias = weights_f_first_moments[i] / (1.0 - pow(beta_one, num_updates));
				second_unbias = weights_f_second_moments[i] / (1.0 - pow(beta_two, num_updates));
				weights_f_gate[i] -= learning_rate * (first_unbias / (sqrt(second_unbias) + 0.0000001));

				// g gate
				weights_g_first_moments[i] = beta_one * weights_g_first_moments[i] + (1.0 - beta_one) * weight_g_dldws[i];
				weights_g_second_moments[i] = beta_two * weights_g_second_moments[i] + (1.0 - beta_two) * weight_g_dldws[i] * weight_g_dldws[i];
				first_unbias = weights_g_first_moments[i] / (1.0 - pow(beta_one, num_updates));
				second_unbias = weights_g_second_moments[i] / (1.0 - pow(beta_two, num_updates));
				weights_g_gate[i] -= learning_rate * (first_unbias / (sqrt(second_unbias) + 0.0000001));
	
				// o gate
				weights_o_first_moments[i] = beta_one * weights_o_first_moments[i] + (1.0 - beta_one) * weight_o_dldws[i];
				weights_o_second_moments[i] = beta_two * weights_o_second_moments[i] + (1.0 - beta_two) * weight_o_dldws[i] * weight_o_dldws[i];
				first_unbias = weights_o_first_moments[i] / (1.0 - pow(beta_one, num_updates));
				second_unbias = weights_o_second_moments[i] / (1.0 - pow(beta_two, num_updates));
				weights_o_gate[i] -= learning_rate * (first_unbias / (sqrt(second_unbias) + 0.0000001));
				} 
	
			// update the biases
			cout<<endl<<"MOMENTUMED UPDATE:\t";
			for (int i = 0; i < int(bias_i_dldbs.size()); i++)
				{
				float first_unbias;
				float second_unbias;
	
				// i gate		
				biases_i_first_moments[i] =  beta_one * biases_i_first_moments[i] + (1.0 - beta_one) * bias_i_dldbs[i];
				biases_i_second_moments[i] = beta_two * biases_i_second_moments[i] + (1.0 - beta_two) * bias_i_dldbs[i] * bias_i_dldbs[i];
				first_unbias = biases_i_first_moments[i] / (1.0 - pow(beta_one, num_updates));
				second_unbias = biases_i_second_moments[i] / (1.0 - pow(beta_two, num_updates));
				biases_i[i] -= learning_rate * (first_unbias / (sqrt(second_unbias) + 0.0000001));

				// f gate		
				biases_f_first_moments[i] =  beta_one * biases_f_first_moments[i] + (1.0 - beta_one) * bias_f_dldbs[i];
				biases_f_second_moments[i] = beta_two * biases_f_second_moments[i] + (1.0 - beta_two) * bias_f_dldbs[i] * bias_f_dldbs[i];
				first_unbias = biases_f_first_moments[i] / (1.0 - pow(beta_one, num_updates));
				second_unbias = biases_f_second_moments[i] / (1.0 - pow(beta_two, num_updates));
				biases_f[i] -= learning_rate * (first_unbias / (sqrt(second_unbias) + 0.0000001));

				// g gate		
				biases_g_first_moments[i] =  beta_one * biases_g_first_moments[i] + (1.0 - beta_one) * bias_g_dldbs[i];
				biases_g_second_moments[i] = beta_two * biases_g_second_moments[i] + (1.0 - beta_two) * bias_g_dldbs[i] * bias_g_dldbs[i];
				first_unbias = biases_g_first_moments[i] / (1.0 - pow(beta_one, num_updates));
				second_unbias = biases_g_second_moments[i] / (1.0 - pow(beta_two, num_updates));
				biases_g[i] -= learning_rate * (first_unbias / (sqrt(second_unbias) + 0.0000001));

				cout<<(learning_rate * (first_unbias / (sqrt(second_unbias) + 0.0000001)))<<"\t";


				// o gate		
				biases_o_first_moments[i] =  beta_one * biases_o_first_moments[i] + (1.0 - beta_one) * bias_o_dldbs[i];
				biases_o_second_moments[i] = beta_two * biases_o_second_moments[i] + (1.0 - beta_two) * bias_o_dldbs[i] * bias_o_dldbs[i];
				first_unbias = biases_o_first_moments[i] / (1.0 - pow(beta_one, num_updates));
				second_unbias = biases_o_second_moments[i] / (1.0 - pow(beta_two, num_updates));
				biases_o[i] -= learning_rate * (first_unbias / (sqrt(second_unbias) + 0.0000001));
				}

			// Reset the gradients.
			for (int i = 0; i < int(output_neurons.size()); i++)
				{
				// output neurons // TO DO: should this be done during backprop?
				output_neurons[i]->set_upstream_gradient(0.0);
				
				// i gate
				bias_i_dldbs[i] = 0.0;
	
				// f gate
				bias_f_dldbs[i] = 0.0;
	
				// g gate
				bias_g_dldbs[i] = 0.0;
	
				// o gate
				bias_o_dldbs[i] = 0.0;

				// cell states and their gradients
				cell_states_dldc[i] = 0.0;
				cell_states[i] = 0.0;

				// set the pre_f_gate_values back to 0.0
				pre_f_gate_values[i] = 0.0; 
		
				// upstream_gradients_recurrent
				upstream_gradients_recurrent[i] = 0.0;
				}	

			for (int i = 0; i < int(weight_i_dldws.size()); i++)
				{
				// i gate
				weight_i_dldws[i] = 0.0;
	
				// f gate
				weight_f_dldws[i] = 0.0;

				// g gate
				weight_g_dldws[i] = 0.0;
			
				// o gate 
				weight_o_dldws[i] = 0.0;
				}

			return;
			}

	};

#endif
