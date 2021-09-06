#include <iostream>
#include <algorithm>
#include <vector>

using namespace std;

vector<int> get_random_order_of_length(int num_indexes)
	{
	vector<int> ret_val;
	for (int i = 0; i < num_indexes; i++) { ret_val.push_back(i); }
	random_shuffle(ret_val.begin(), ret_val.end());
	return ret_val;
	}
