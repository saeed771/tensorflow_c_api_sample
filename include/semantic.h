#include "tf_model.h"

class Semantic_Seg
{
  public:
    Semantic_Seg();
    Semantic_Seg(int h, int w, int c, std::string model_name, std::string input_tensors, std::vector<std::string> output_tensor_names);
    ~Semantic_Seg();

    int get_mask(unsigned char *new_image, unsigned char *binary_mask);
TF_Model *model;
  private:
    int height;
    int width;
    int channels;
    int size;
    int res;
    float *mask_buff;
    
    // get Next Frame
    int update(unsigned char *new_image);
};