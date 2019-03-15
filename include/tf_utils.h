#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include "c_api.h"
#include <numeric>
const char *_tf_data_type_to_string(TF_DataType data_type);
void _print_op_inputs(std::ofstream &modelinfo, TF_Operation *op);
void _print_op_outputs(std::ofstream &modelinfo, TF_Graph *graph, TF_Operation *op);
void print_tf_version();
void _print_tn_info(TF_Graph *graph, const char *layer_name);
void tf_freeBuffer(void *data, size_t length);
void tf_deallocator(void *ptr, size_t len, void *arg);