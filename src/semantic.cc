#include "semantic.h"
Semantic_Seg::Semantic_Seg(int h, int w, int c,
                           std::string model_fpath,
                           std::string input_tn_name,
                           std::vector<std::string> output_tns_names)
{
    this->channels = c;
    this->height = h;
    this->width = w;
    this->size = height * width * channels;
    this->res = this->height * this->width;
    std::vector<int64_t> input_dimensions = {1, h, w, c};
    this->model = new TF_Model();
    this->model->set_input_dims(input_dimensions);
    this->model->load_model(model_fpath);
    this->model->set_tns_names(input_tn_name, output_tns_names, false);
    this->model->set_session();
}
Semantic_Seg::~Semantic_Seg()
{
}
int Semantic_Seg::update(unsigned char *image)
{
    int r = this->model->run(image);
    return r;
}

int Semantic_Seg::get_mask(unsigned char *image, unsigned char *binary_mask)
{

    this->update(image);

    std::vector<TF_Tensor *> output_tns = this->model->get_output_tns();
    unsigned char *x = static_cast<unsigned char *>(TF_TensorData(output_tns[0]));
    for (int i = 0; i < this->res; ++i)
    {
        if (x[i] > 0.1)
        {
            binary_mask[i] = 255;
        }
        else
        {
            binary_mask[i] = 0;
        }
    }
    // free(x);
    return 0;
}
