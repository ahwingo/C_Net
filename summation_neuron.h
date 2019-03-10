#include <iostream>
#include <vector>
#include "math.h"

using namespace std;

#ifndef __SUMMATION_NEURON_INCLUDED__
#define __SUMMATION_NEURON_INCLUDED__

//////////////////////////////////////////////////////////////////////////////////////////////////////
//      Summation (Post Convolutional / Image)  Neuron Class Definition
///////////////////////////////////////////////////////////////////////////////////////////////////////

class summation_neuron: public neuron
        {

        public:

                int index_of_corresponding_output;


                summation_neuron(int ioco) : neuron(0.0, 0.0, 0.0, 0.0, NULL, NULL)
                        {
                        index_of_corresponding_output = ioco;
                        }


                void add_input_neuron(neuron * new_input)
                        {
                        new_input->add_ouput_neuron(this);
                        input_neurons.push_back(new_input);
                        }


                void back_prop()
                        {
                        // Pass the upstream gradient at this neuron to each of its input neurons.
                        for (int i = 0; i < int(input_neurons.size()); i++)
                                {
                                // Cast the input neuron to a conv_filter.
                                conv_filter_2D * an_input= (conv_filter_2D *)input_neurons[i];
                                an_input->increment_upstream_gradient_at_index(upstream_gradient, index_of_corresponding_output);
                                }

                        // Reset the upstream gradient to zero.
                        upstream_gradient = 0.0;
                        }


                void forward_prop()
                        {
                        // Reset the output to zero.
                        output = 0.0;

                        // Increment the reset output with each input value.
                        for (int i = 0; i < int(input_neurons.size()); i++)
                                {
                                // Cast the input neuron to a conv_filter.
                                conv_filter_2D * an_input = (conv_filter_2D *)input_neurons[i];
                                output += an_input->get_output_at_index(index_of_corresponding_output);
                                }
                        }
        };

#endif
