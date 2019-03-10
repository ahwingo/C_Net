#include <iostream>
#include <fstream>
#include <string>
#include <vector>

using namespace std;

vector<float> get_values_from_tsv_string(string line)
	{
	vector<float> ret_val;
	string delimiter = "\t";
	int pos = 0;
	float value;
	string remaining_string = line;
	while (remaining_string.find(delimiter) != string::npos)
		{
		value = stof(remaining_string.substr(0, remaining_string.find(delimiter)));
		ret_val.push_back(value);
		int pos_of_item_after_delimiter = remaining_string.find(delimiter) + delimiter.length();
		remaining_string = remaining_string.substr(pos_of_item_after_delimiter, remaining_string.length());
		}
	// Make sure to add the last column.
	value = stof(remaining_string);
	ret_val.push_back(value);
	return ret_val;
	}


vector< vector<float> > load_data_from_tsv_file(string filename)
	{
	vector< vector<float> > ret_val;
	ifstream the_file(filename);
	string line;
	if (the_file.is_open())
		{
		while (getline(the_file, line))
			{
			vector<float> values = get_values_from_tsv_string(line);
			ret_val.push_back(values);
			}
		the_file.close();
		}
	return ret_val;
	}
