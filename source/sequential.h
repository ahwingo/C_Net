#include <iostream>
#include <random>
#include "assert.h"
#include "neuron.h"
#include "conv_filter_2D.h"
#include "summation_neuron.h"
#include "activation_functions.h"
#include "cost_functions.h"
#include "random_indexer.h"

using namespace std;

#ifndef __SEQUENTIAL_INCLUDED__
#define __SEQUENTIAL_INCLUDED__

class sequential_network
	{

	private:

		vector< vector<neuron *> > list_of_all_layers;		// Vector to store all of the layers of the network.
		vector<string> layer_types;				// List that keeps up with layer types: "simple", "conv_2D", "summation", "lstm" etc.

		float learning_rate;					// The networks learning rate.
		float beta_one;						// ADAM optimization param.
		float beta_two;						// ADAM optimization param.

		normal_distribution<float> distribution;		// Distribution used to initialize network parameters.
		default_random_engine generator;			// Random generator used to initialize network parameters.

		float (*loss_func) (vector<neuron *>, vector<float>);	// The loss function used to train the sequential network.
		
	public:

		// The initializer for a sequential network.
		// Inputs include:
		//	lr: the learning rate.
		//	b1: ADAM optimization param.
		//	b2: ADAM optimization param.	
		//	dist: distribution used for weight initialization.
		//	l_func: the loss function that will be used to train the network.
		sequential_network(float lr, float b1, float b2, normal_distribution<float> dist, float (*l_func) (vector<neuron *>, vector<float>))
			{
			learning_rate = lr;
			beta_one = b1;
			beta_two = b2;
			distribution = dist;
			loss_func = l_func;
			}


		// This function adds the first layer to the network.
		// Returns the added layer.
		// Inputs include:
		//	layer_size: size of layer to return.
		vector<neuron *> add_input_layer(int layer_size)
			{
			assert(int(list_of_all_layers.size()) == 0);	// There cannot already be an input layers.		
			vector<neuron *> layer_to_add;		
			for (int i = 0; i < layer_size; i++)
				{
				neuron * n = new neuron(0.0, 0.0, 0.0, 0.0, NULL, NULL);
				layer_to_add.push_back(n);
				}
			list_of_all_layers.push_back(layer_to_add);
			layer_types.push_back("simple");	
			return layer_to_add;
			}

		
		// This function adds a simple fully connected layer to the network.
		// Returns the added layer.
		// Inputs include:
		//	layer_size: size of layer to return.
		//	f: activation funciton for the neuron.
		//	df: derivative of neurons activation funciton.
		vector<neuron *> add_fully_connected_layer(int layer_size, float (*f) (float), float (*df) (float))
			{
			assert(int(list_of_all_layers.size()) > 0);	// There must at least already be an input layer.
			vector<neuron *> layer_to_add;
			for (int i = 0; i < layer_size; i++)
				{
				float bias = distribution(generator);
				neuron * n = new neuron(bias, learning_rate, beta_one, beta_two, f, df);
				// Connect the new neuron to the previous layer.
				vector<neuron *> prev_layer = list_of_all_layers.back();
				for (int j = 0; j < int(prev_layer.size()); j++)
					{
					n->add_input_neuron_with_weight(prev_layer[j], distribution(generator));
					}
				layer_to_add.push_back(n);
				}
			list_of_all_layers.push_back(layer_to_add);
			layer_types.push_back("simple");	
			return layer_to_add;
			}


		// This function adds a 2D convolutional layer to the network, followed by an activation layer with the given type.
		// NOTE TO SELF:
		//	Clearly, you need to rewrite the conv layer class to simplify this. 
		//	It should set up its own output layer, like the lstm does.
		// Returns the following activation layer.
		// Inputs include:
		//	kernel_width: the length of the side of the square kernel.
		//	input_width: the length of the side of a square input image.		
		//	stride: the kernels stride.
		//	use_padding: whether or not the convolution will use padding.
		//	f: followingl layer activation.
		//	df: following layer activation derivative.
		vector<neuron *> add_2D_convolutional_layer_with_following_activation(int kernel_width, int input_width, int stride, bool use_padding, float (*f) (float), float (*df) (float))
			{
			assert(int(list_of_all_layers.size()) > 0);	// There must at least already be an input layer.
			vector<neuron *> conv_layer_to_add;
			vector<float> weights;
			for (int i = 0; i < kernel_width*kernel_width; i++)
				{
				weights.push_back(distribution(generator));
				}
			float bias = distribution(generator);
			conv_filter_2D * conv_layer = new conv_filter_2D(weights, bias, learning_rate, beta_one, beta_two, input_width, stride, use_padding);
			// Connect the conv layer to the previous layer.
			vector<neuron *> prev_layer = list_of_all_layers.back();
			for (int i = 0; i < int(prev_layer.size()); i++)
				{
				conv_layer->add_input_neuron(prev_layer[i]);
				}
			conv_layer_to_add.push_back(conv_layer);	
			list_of_all_layers.push_back(conv_layer_to_add);
			layer_types.push_back("conv_2D");
			// Create the summation layer to add after the conv 2D layer.
			vector<neuron *> summ_layer_to_add;
			for (int i = 0; i < (input_width - kernel_width + 1)*(input_width - kernel_width + 1); i++)
				{
				summation_neuron * s_n = new summation_neuron(i);
				s_n->add_input_neuron(conv_layer);
				summ_layer_to_add.push_back(s_n);
				}
			list_of_all_layers.push_back(summ_layer_to_add);
			layer_types.push_back("summation");
			// Create the activation layer that follows the fully connected layer.
			vector<neuron *> activation_layer_to_add;
			for (int i = 0; i < int(summ_layer_to_add.size()); i++)
				{
				float bias = distribution(generator); 
				neuron * n = new neuron(bias, learning_rate, beta_one, beta_two, f, df);
				float weight = distribution(generator);
				n->add_input_neuron_with_weight(summ_layer_to_add[i], weight);
				activation_layer_to_add.push_back(n);
				}			
			list_of_all_layers.push_back(activation_layer_to_add);
			layer_types.push_back("simple");
			return activation_layer_to_add;
			}


		// This function sets the input to the network, by modifying the output values of the input layer.
		// Inputs include:
		//	input_values: the vector of input values to train / test / validate with.
		void set_input_values_of_network(vector<float> input_values)
			{
			//cout<<"set_input_values_of_network()"<<endl;
			// The call to this function will fail if the sizes of the two input vectors are not equal.
			vector<neuron *> input_layer = list_of_all_layers.front();
			assert(input_layer.size() == input_values.size());
			for (int i = 0; i < int(input_layer.size()); i++)
				{
				input_layer[i]->set_output(input_values[i]);
				}	
	

			// check to make sure the input is correctly set up.
//			for (int i = 0; i < int(list_of_all_layers[0].size()); i++)
//				{
//				cout<<"input value at index "<<i<<" is "<<list_of_all_layers[0][i]->get_output()<<endl;
//				}

			return;
			}


		// This function calculates the loss at an layer given a loss function and the correct output.
		// It returns a tuple holding the { loss, output label index, actual label index }
		// Inputs include:
		//	final_layer: the final layer of a network, where loss is calculated.
		//	loss_func: the loss function used.
		//	correct_output: the output that the network should have generated.
		vector<float> calculate_loss(vector<float> correct_output)
			{
			assert(layer_types.back() == "simple");	// The final layer has to be a simple fully connected layer.
			vector<neuron *> final_layer = list_of_all_layers.back();
			float the_total_loss = loss_func(final_layer, correct_output);
			int actual_index = 0;
			for (int i = 0; i < int(correct_output.size()); i++) { if (correct_output[i] > 0) actual_index = i; }
			float max_output = final_layer.front()->get_output();
			int output_index = 0;
			for (int i = 0; i < int(final_layer.size()); i++) 
				{ 
				if (final_layer[i]->get_output() > max_output) 
					{
					max_output = final_layer[i]->get_output();
					output_index = i;
					}
				}
			vector<float> ret_val;
			ret_val.push_back(the_total_loss);
			ret_val.push_back(float(output_index));
			ret_val.push_back(float(actual_index));
			return (ret_val);
			}

		
		// This function does a forward pass on the network, with whatever input value is currently set (by set_input_values_of_network()).
		// Returns the output of the final layer, as a vector of floats.
		vector<float> run_forward_prop_pass()
			{
			// Call forward_prop on every layer after the input layer. 
			for (int i = 1; i < int(list_of_all_layers.size()); i++)
				{
				vector<neuron *> the_layer = list_of_all_layers[i];
				for (int j = 0; j < int(the_layer.size()); j++)
					{
					the_layer[j]->forward_prop();
					}
				}
			//cout<<"f prop done"<<endl;
			// Now, return the output of the network as a vector.
			vector<neuron *> final_layer = list_of_all_layers.back();
			vector<float> ret_val;
			for (int i = 0; i < int(final_layer.size()); i++)
				{
				ret_val.push_back(final_layer[i]->get_output());
				}
			return ret_val;
			}


		// This function runs back prop on the network, updating the parameters layer by layer. 
		// It assumes the loss has already been calculated with calculate_loss().
		void run_back_prop_pass()
			{
			// Call back_prop on every layer from the final layer to the first hidden layer, inclusive.
			for (int i = int(list_of_all_layers.size()) - 1; i > 0; i--)
				{
				vector<neuron *> the_layer = list_of_all_layers[i];
				for (int j = 0; j < int(the_layer.size()); j++)
					{
					the_layer[j]->back_prop();
					}
				}
			return;
			}


		// This function trains and evaluates the network with the given data over a number of epochs.
		// Returns a vector of the average loss at each epoch.
		// Inputs include:
		//	num_epochs: the number of epochs to train for.
		//	t_inputs: the vector of input values (vectors) for training.
		//	t_labels: the vector of output values (vectors) for training.
		//	e_inputs: the vector of input values (vectors) for evaluation.
		//	e_labels: the vector of output values (vectors) for evaluation.
		vector<float> train_network_for_epochs_with_data(int num_epochs, vector< vector<float> > t_inputs, vector< vector<float> > t_labels,  
			vector< vector<float> > e_inputs, vector< vector<float> > e_labels)
			{
			vector <float> ret_val;
			for (int i = 1; i <= num_epochs; i++)
				{
				// Index over the training data in a random order.
				vector<int> random_indexing_array = get_random_order_of_length(t_inputs.size());

				// Train the network over each training instance.
				for (int j = 0; j < int(t_inputs.size()); j++)
					{
					// Use the random index.
					int rand_index = random_indexing_array[j];

					// Set the input to the network.
					set_input_values_of_network(t_inputs[rand_index]);
		
					// Forward propagation step.
					run_forward_prop_pass();
				
					// Calculate the loss.
					calculate_loss(t_labels[rand_index])[0];

					// Back propagation step.
					run_back_prop_pass();		
					}

				// Evaluate the network. Calculate the average loss over the evaluation data.
				// Also calculate the precision, recall, and F1 score.
				float sum_of_each_steps_loss = 0.0;
				int true_positives = 0;
				int false_positives = 0;
				int true_negatives = 0;
				int false_negatives = 0;
				for (int j = 0; j < int(e_inputs.size()); j++)
					{
					// Set the input to the network.
					set_input_values_of_network(e_inputs[j]);
		
					// Forward propagation step.
					run_forward_prop_pass();
	
					// Calculate the loss and add it to the sum.
					vector<float> loss_ret_val = calculate_loss(e_labels[j]);
					int predicted_label = int(loss_ret_val[1]);
					int actual_label = int(loss_ret_val[2]);

					if (predicted_label == 0 && actual_label == 0) true_positives++;
					else if (predicted_label == 0 && actual_label != 0) false_positives++;
					else if (predicted_label == 1 && actual_label == 1) true_negatives++;
					else if (predicted_label == 1 && actual_label != 1) false_negatives++;
					sum_of_each_steps_loss += loss_ret_val[0];
					}
	
				float precision = float(true_positives) / float(true_positives + false_positives);
				float recall = float(true_positives) / float(true_positives + false_negatives);
				float F1 = 2.0 * precision * recall / (precision + recall);
				float avg_loss = sum_of_each_steps_loss/float(e_inputs.size());
				ret_val.push_back(avg_loss);
				cout<<"--------------------------------------------------------"<<endl;
				cout<<"|    EPOCH:     "<<i<<endl;
				cout<<"|    AVG LOSS:  "<<avg_loss<<endl;
				cout<<"|"<<endl;
				cout<<"|    *    A\t\tB  <--- pred"<<endl;
				cout<<"|    A   "<<true_positives<<"\t\t"<<false_negatives<<endl;        
				cout<<"|    B   "<<false_positives<<"\t\t"<<true_negatives<<endl;      
				cout<<"|"<<endl;
				cout<<"|    PRECISION: "<<precision<<endl;
				cout<<"|    RECALL:    "<<recall<<endl;
				cout<<"|    F1:        "<<F1<<endl;
				cout<<"--------------------------------------------------------"<<endl;
				}

			return ret_val;
			} 


		// print output layers output
		void print_network_output()
			{
			vector<neuron *> final_layer = list_of_all_layers.back();
			for (int i = 0; i < int(final_layer.size()); i++)
				{
				cout<<"OUTPUT INDEX: "<<i<<"\t\tOUTPUT VALUE: "<<final_layer[i]->get_output()<<endl;
				}
			}


		int number_of_layers_in_network() { return int(list_of_all_layers.size()); }


	};

#endif
