#include <iostream>
#include <vector>
#include "math.h"

using namespace std;

#ifndef __NEURON_INCLUDED__
#define __NEURON_INCLUDED__

///////////////////////////////////////////////////////////////////////////////////////////////////////
//	Super Neuron Class Definition
///////////////////////////////////////////////////////////////////////////////////////////////////////

class neuron
	{

	public:

		vector<float>  weights;				// neuron's weights,	R^N
		vector<neuron *> input_neurons;			// dendrites,		R^N

		float bias;					// neuron's bias
		float learning_rate;				// neuron's learning rate
		
		float beta_one;					// param for ADAM optimization
		float beta_two;					// param for ADAM optimization	
		float num_updates;				// param for ADAM optimization

		float bias_first_moment;			// param for ADAM optimization
		float bias_second_moment;			// param for ADAM optimization

		vector<float> weights_first_moments;		// param for ADAM optimization
		vector<float> weights_second_moments;		// param for ADAM optimization

		vector<float> inputs_first_moments;		// param for ADAM optimization
		vector<float> inputs_second_moments;		// param for ADAM optimization

		float (*activation_func) (float);		// function pointer to the activation function
		float (*deriv_act_func) (float);		// function pointer to the derivative of the activation function 

		float output;					// neuron's output 
		float weighted_input;				// input to activation function
		vector<neuron *> output_neurons;		// axon terminals
	
		float upstream_gradient;			// neuron's upstream gradient

		int number_of_forward_passes;			// counts the number of times forward_prop() has been called

		neuron(float b, float lr, float b1, float b2, float (*f) (float), float (*df) (float))
			{
		
			/* 
			 * A neuron is initialized with:
			 *	a bias,
			 *	a learning rate,
			 *	a value for beta_one (ADAM optimization)
			 *	a value for beta_two (ADAM optimization)
			 *	a function pointer to the activation function
			 *	a function pointer to the derivative of the activation function
			 */

			bias = b;				// weights and bias initialized by network
	
			learning_rate = lr;			// learning rate set by the network
			beta_one = b1;				// also set by the network, used for ADAM
			beta_two = b2;				// also set by the network, used for ADAM
			num_updates = 0;

			activation_func = f;
			deriv_act_func = df;

			bias_first_moment = 0.0;		// used for ADAM
			bias_second_moment = 0.0;		// used for ADAM

			output = 0.0;
			weighted_input = 0.0;
			upstream_gradient = 0.0;		// updated by the output neurons during back_prop

			number_of_forward_passes = 0;

			}

		void add_input_neuron_with_weight(neuron * new_input, float weight)
			{
			new_input->add_ouput_neuron(this);
			input_neurons.push_back(new_input);
			weights.push_back(weight);
			weights_first_moments.push_back(0.0);
			weights_second_moments.push_back(0.0);	
			inputs_first_moments.push_back(0.0);
			inputs_second_moments.push_back(0.0);
			}

	
		// one line helper functions
		float get_output() { return output; }
		void set_output(float new_output) { output = new_output; } 
		void add_ouput_neuron(neuron * new_output) { output_neurons.push_back(new_output); }
		void increment_upstream_gradient(float value) { upstream_gradient += value; }
		void set_upstream_gradient(float value) { upstream_gradient = value; }
		float get_upstream_gradient() { return upstream_gradient; }
			

		void display_weights_and_bias()
			{
			cout<<"W = [ ";
			for (int i = 0; i < int(weights.size()); i++) { cout<<weights[i]<<" "; }
			cout<<"]"<<endl; 
			cout<<"B = "<<bias<<endl<<endl;
			}
	


		void update_bias_with_upstream_gradient(float lg)
			{
			bias_first_moment = beta_one * bias_first_moment + (1.0 - beta_one) * lg;
			bias_second_moment = beta_two * bias_second_moment + (1.0 - beta_two) * lg * lg;
			float first_unbias = bias_first_moment / (1.0 - pow(beta_one, num_updates));
			float second_unbias = bias_second_moment / (1.0 - pow(beta_two, num_updates));
			float db = first_unbias / (sqrt(second_unbias) + 0.0000001);
			bias -= learning_rate * db;
			}


		void update_weights_with_upstream_gradient(float lg)
			{
			for (int i = 0; i < int(weights.size()); i++)
				{
				float dwi = lg * input_neurons[i]->get_output();
				weights_first_moments[i] = beta_one * weights_first_moments[i] + (1.0 - beta_one) * dwi;
				weights_second_moments[i] = beta_two * weights_second_moments[i] + (1.0 - beta_two) * dwi * dwi;
				float first_unbias = weights_first_moments[i] / (1.0 - pow(beta_one, num_updates));
				float second_unbias = weights_second_moments[i] / (1.0 - pow(beta_two, num_updates));
				dwi = first_unbias / (sqrt(second_unbias) + 0.0000001);
				weights[i] -= learning_rate * dwi;
				}
			}


		void update_upstream_gradient_of_input_neurons_using(float lg)
			{
			for (int i = 0; i < int(input_neurons.size()); i++)
				{
				float dxi = lg * weights[i];
				input_neurons[i]->increment_upstream_gradient(dxi);
				}
			}


		void back_prop()
			{
			// increment the number of updates
			num_updates += 1.0;

			// calculate the local gradient of the activation function
			// note that this gradient value is upstream of the weighted inputs and bias
			float local_grad = deriv_act_func(weighted_input)*upstream_gradient;
			
			// update the bias
			update_bias_with_upstream_gradient(local_grad);

			// update the weights
			update_weights_with_upstream_gradient(local_grad);
			
			// update the upstream gradients of the input neurons
			update_upstream_gradient_of_input_neurons_using(local_grad);

			// reset this neuron's upstream gradient back to zero
			upstream_gradient = 0.0;
			}


		void forward_prop()
			{
			// increment the count of forward prop calls
			number_of_forward_passes += 1;

			// reset the weighted input to zero
			float weighted_input = 0.0;
			weighted_input = 0.0;

			// also reset the upstream gradient to zero?
			upstream_gradient = 0.0;
			
			// calculate the sum of the bias and the dot product of input_neurons and weights
			for (int i = 0; i < int(input_neurons.size()); i++)
				{
				weighted_input += input_neurons[i]->get_output()*weights[i];
				}
			weighted_input += bias;

			// pass the weighted input to the activation function and set value of output
			output = activation_func(weighted_input);
			}	
	};

#endif
