#include <iostream>
#include "math.h"

using namespace std;

#ifndef __COST_FUNCS_INCLUDED__
#define __COST_FUNCS_INCLUDED__


//----------------------------------------------------------------------------------------------
//	 softmax loss function
//----------------------------------------------------------------------------------------------
/*
*	recall that with softmax, there is only one correct output (ie one-hot encodings).
*	each output layer neuron represents a unique class.
*
*	the vector returned holds upstream gradient values for each output neuron.
*/
//----------------------------------------------------------------------------------------------

float softmax_loss(vector<neuron *> output_layer_neurons, vector<float> correct_output)
	{
	// get the index of the correct one-hot output	
	int index_of_correct_output = 0;	
	for (int i = 0; i < int(correct_output.size()); i++)
		{
		if (correct_output[i] > 0) 
			{
			index_of_correct_output = i;
			break;
			}
		}

	// the first step is to get the max of the scores, 
	// and then subtract the max from each score (make 0 the highest output)
	float max_score = output_layer_neurons[0]->get_output();
	for (int i = 0; i < output_layer_neurons.size(); i++)
		{
		if (output_layer_neurons[i]->get_output() > max_score) 
			{ 
			max_score = output_layer_neurons[i]->get_output(); 
			}
		}
	vector<float> treated_scores;
	for (int i = 0; i < output_layer_neurons.size(); i++)
		{
		treated_scores.push_back(output_layer_neurons[i]->get_output() - max_score);
		}


	// the second step is to exponentiate and keep track of the sum
	float sum_of_exp_scores = 0.0;
	for (int i = 0; i < treated_scores.size(); i++)
		{
		float exp_score = exp(treated_scores[i]); 
		sum_of_exp_scores += exp_score;
		treated_scores[i] = exp_score;
		}	

	// the third step is to calculate the loss
	float loss = -log(treated_scores[index_of_correct_output] / sum_of_exp_scores);

	// finally, find the upstream gradient for the loss at each output neuron, update and return
	for (int i = 0; i < output_layer_neurons.size(); i++)
		{
		if (i == index_of_correct_output)
			{
			float grad = (treated_scores[index_of_correct_output] / sum_of_exp_scores) - 1.0;
			output_layer_neurons[i]->increment_upstream_gradient(grad);
			}
		else
			{
			float grad = (treated_scores[i] / sum_of_exp_scores);
			output_layer_neurons[i]->increment_upstream_gradient(grad);
			}

		}

	return loss;
	}


#endif
