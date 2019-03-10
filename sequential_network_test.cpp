#include <iostream>
#include <random>
#include <tuple>
#include "neuron.h"
#include "sequential.h"
#include "activation_functions.h"
#include "cost_functions.h"
#include "tsv_data_loader.h"

using namespace std;

int main()
	{

	//------------------------------------------------------------------------------------------------------------------------
	// Load the training and evaluation data from the tsv file. 
	// The 'circles' and 'moons' datasets both have input / output dimensions of 2.
	//------------------------------------------------------------------------------------------------------------------------
	vector< vector<float> > training_inputs = load_data_from_tsv_file("circles_training_inputs.txt");
	vector< vector<float> > training_labels = load_data_from_tsv_file("circles_training_labels.txt");
	vector< vector<float> > evaluation_inputs = load_data_from_tsv_file("circles_evaluation_inputs.txt");
	vector< vector<float> > evaluation_labels = load_data_from_tsv_file("circles_evaluation_labels.txt");


	//------------------------------------------------------------------------------------------------------------------------
	// Set hyperparameters.
	//------------------------------------------------------------------------------------------------------------------------
	int num_epochs = 1000;
	float learning_rate = 0.001;
        float beta_one = 0.9;
        float beta_two = 0.999;
	normal_distribution<float> distribution(0.0, 1.0);
	int input_layer_size = int(training_inputs[0].size());		
	int output_layer_size = int(training_labels[0].size()); 


	//------------------------------------------------------------------------------------------------------------------------
	// Create a sequential network.
	//------------------------------------------------------------------------------------------------------------------------
	
	// Initialize the network.
	sequential_network the_network = sequential_network(learning_rate, beta_one, beta_two, distribution, softmax_loss);

	// Set up the networks input layer.
	the_network.add_input_layer(input_layer_size);

	// Add a few fully connected layers, each containing a number of hidden neurons. Use a sigmoid activation.
	the_network.add_fully_connected_layer(16, sigmoid, sigmoid_derivative);
	the_network.add_fully_connected_layer(4, sigmoid, sigmoid_derivative);
	the_network.add_fully_connected_layer(8, sigmoid, sigmoid_derivative);

	// Add the output layer. Use a sigmoid activation.
	the_network.add_fully_connected_layer(output_layer_size, sigmoid, sigmoid_derivative);

	
	//------------------------------------------------------------------------------------------------------------------------
	// Train the network for a number of epochs. Test accuracy at end of each epoch.
	//------------------------------------------------------------------------------------------------------------------------
	the_network.train_network_for_epochs_with_data(num_epochs, training_inputs, training_labels, evaluation_inputs, evaluation_labels);




/*
	for (int i = 1; i <= num_epochs; i++)
		{
		// Index over the training data in a random order.
		vector<int> random_indexing_array = get_random_order_of_length(training_inputs.size());

		// Train the network over each training instance.
		for (int j = 0; j < int(training_inputs.size()); j++)
			{
			// Use the random index.
			int rand_index = random_indexing_array[j];

			// Set the input to the network.
			the_network.set_input_values_of_network(training_inputs[rand_index]);
		
			// Forward propagation step.
			the_network.run_forward_prop_pass();
			
			// Calculate the loss.
			the_network.calculate_loss(training_labels[rand_index])[0];

			// Back propagation step.
			the_network.run_back_prop_pass();		
			}

		// Evaluate the network. Calculate the average loss over the evaluation data.
		// Also calculate the precision, recall, and F1 score.
		float sum_of_each_steps_loss = 0.0;
		int true_positives = 0;
		int false_positives = 0;
		int true_negatives = 0;
		int false_negatives = 0;
		for (int j = 0; j < int(evaluation_inputs.size()); j++)
			{
			// Set the input to the network.
			the_network.set_input_values_of_network(evaluation_inputs[j]);
		
			// Forward propagation step.
			the_network.run_forward_prop_pass();
	
			// Calculate the loss and add it to the sum.
			vector<float> loss_ret_val = the_network.calculate_loss(evaluation_labels[j]);
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
		float avg_loss = sum_of_each_steps_loss/float(evaluation_inputs.size());
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

*/

	return 0;
	}
