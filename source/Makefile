CC = g++
CFLAGS  = -g -Wall

default: test

test:  sequential_network_test.o
	$(CC) $(CFLAGS) -o test sequential_network_test.o

sequential_network_test.o:  tsv_data_loader.h sequential_network_test.cpp summation_neuron.h sequential.h random_indexer.h neuron.h lstm.h cost_functions.h conv_filter_2D.h activation_functions.h
	$(CC) $(CFLAGS) -c sequential_network_test.cpp

clean:
	$(RM) test *.o *~