
#include "tf_utils.h"
#ifndef __cplusplus
#error tf_model.h header must be compiled as C++11
#endif
class TF_Model
{
public:
  TF_Model();
  TF_Model(std::vector<int64_t> input_Dimensions);
  ~TF_Model();
  //
  void set_tns_names(std::string input_tn_name,
                           std::vector<std::string> output_tn_names,
                           bool inspect = false);
  void set_input_dims(std::vector<int64_t> input_Dimensions);
  int get_data_size();
  std::vector<TF_Tensor *> get_output_tns();
  std::vector<int64_t> get_input_dims();

  int load_model(const std::string model_pb_file);
  int set_session();
  int run(unsigned char *image);

  void inspect(const std::string output_file_name);

private:
  TF_Graph *graph;
  TF_Status *status;
  TF_Session *sess;
  TF_Tensor *input_tn;
  TF_Output input_op;
  std::vector<TF_Output> out_ops;
  std::vector<TF_Tensor *> output_tns;
  size_t data_size;
  std::string input_tn_name;
  std::vector<int64_t> input_dims;
  std::vector<std::string> output_tn_names;
};