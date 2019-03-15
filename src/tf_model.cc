#include "tf_model.h"
#include <fstream>

TF_Model::TF_Model()
{
    this->graph = TF_NewGraph();
    this->status = TF_NewStatus();
    this->data_size = 0;
}

TF_Model::TF_Model(std::vector<int64_t> input_dimensions)
{
    this->graph = TF_NewGraph();
    this->status = TF_NewStatus();
    this->input_dims = input_dimensions;
    this->data_size = std::accumulate(std::begin(input_dims),
                                      std::end(input_dims),
                                      1,
                                      std::multiplies<int64_t>());
}

TF_Model::~TF_Model()
{
    // free all the tensors
    TF_DeleteStatus(status);
    TF_DeleteGraph(graph);
}

void TF_Model::set_tns_names(std::string input_tensor_name, std::vector<std::string> output_tensor_names, bool inspect)
{
    this->input_tn_name = input_tensor_name;
    this->output_tn_names = output_tensor_names;

    if (inspect)
    {
        int num_output_tensors = output_tensor_names.size();
        _print_tn_info(this->graph, this->input_tn_name.data());
        for (int i = 0; i < num_output_tensors; ++i)
        {
            _print_tn_info(this->graph, this->output_tn_names[i].data());
        }
    }
}

void TF_Model::set_input_dims(std::vector<int64_t> input_Dimensions)
{
    this->input_dims = input_Dimensions;
    this->data_size = std::accumulate(std::begin(input_dims), std::end(input_dims), 1, std::multiplies<int64_t>());
}

int TF_Model::get_data_size()
{
    return this->data_size;
}
std::vector<TF_Tensor *> TF_Model::get_output_tns()
{
    return this->output_tns;
}

std::vector<int64_t> TF_Model::get_input_dims()
{
    return this->input_dims;
}

int TF_Model::load_model(const std::string model_pb_file)
{
    //open the protobuf file
    FILE *f = fopen(model_pb_file.data(), "rb");
    if (f == nullptr)
    {
        return -1;
    }
    fseek(f, 0, SEEK_END);
    //get the size of the model file
    const auto fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (fsize < 1)
    {
        fclose(f);
        return -1;
    }
    //allocate memory for the protobuf data
    const auto data = malloc(fsize);
    //read the file into memory
    fread(data, fsize, 1, f);
    //close the file
    fclose(f);
    //allocate a protocol buffer empty
    TF_Buffer *buf = TF_NewBuffer();
    //load the protocol buffer with data from memory
    buf->data = data;
    //load the proto buf size
    buf->length = fsize;
    //method to free the memory
    buf->data_deallocator = tf_freeBuffer;
    // TF_ImportGraphDefOptions holds options that can be passed to TF_GraphImportGraphDef.
    TF_ImportGraphDefOptions *opts = TF_NewImportGraphDefOptions();
    // Import the graph serialized in `buffer` into `graph`.
    // Convenience function for when no results are needed.
    TF_GraphImportGraphDef(graph, buf, opts, status);
    //delete options passed to graph
    TF_DeleteImportGraphDefOptions(opts);
    // free the memory
    TF_DeleteBuffer(buf);
    //read status error
    if (TF_GetCode(status) != TF_OK)
    {
        TF_DeleteStatus(status);
        TF_DeleteGraph(graph);
        return -1;
    }
    std::cout<<"Model Loading: Successful\n";
    return 0;
}

int TF_Model::set_session()
{
    TF_SessionOptions *options = TF_NewSessionOptions();
    this->sess = TF_NewSession(this->graph, options, this->status);
    TF_DeleteSessionOptions(options);
    this->input_op = {TF_GraphOperationByName(this->graph, this->input_tn_name.data()), 0};
    for (auto i = 0; i < this->output_tn_names.size(); ++i)
        this->out_ops.push_back({TF_GraphOperationByName(this->graph, this->output_tn_names[i].data()), 0});

    std::vector<int64_t> out_dimensions = {1};
    size_t datasize0 = sizeof(float);

    for (auto i = 0; i < this->output_tn_names.size(); ++i)
        this->output_tns.push_back(TF_AllocateTensor(TF_FLOAT, out_dimensions.data(), 1, datasize0 * 1000000));
    std::cout<<"Session creation: Successful\n";
    return 0;
}

int TF_Model::run(unsigned char *image)
{
    this->input_tn = TF_NewTensor(TF_UINT8,
                                  this->input_dims.data(), this->input_dims.size(),
                                  image, this->data_size,
                                  tf_deallocator, nullptr);
    TF_SessionRun(this->sess,
                  nullptr,                                                             // Run options.
                  &this->input_op, &this->input_tn, 1,                                 // Input tensors, input tensor values, number of inputs.
                  this->out_ops.data(), this->output_tns.data(), this->out_ops.size(), // Output tensors, output tensor values, number of outputs.
                  nullptr, 0,                                                          // Target operations, number of targets.
                  nullptr,                                                             // Run metadata.
                  this->status                                                         // Output status.
    );
    if (TF_GetCode(this->status) != TF_OK)
    {
        std::cout << "Model Error: error in TF_SessionRun\n";
        return -1;
    }

    return 0;
}

void TF_Model::inspect(const std::string output_file_name)
{
    //tensorflow operation
    TF_Operation *op;
    size_t pos = 0;
    std::ofstream model_info;
    model_info.open(output_file_name.data());

    while ((op = TF_GraphNextOperation(this->graph, &pos)) != nullptr)
    {
        const char *name = TF_OperationName(op);
        const char *type = TF_OperationOpType(op);
        const char *device = TF_OperationDevice(op);

        const int num_outputs = TF_OperationNumOutputs(op);
        const int num_inputs = TF_OperationNumInputs(op);
        model_info << "==========================================" << std::endl;
        model_info << "operation position: " << pos << " : " << std::endl;
        model_info << "\t name:" << name << std::endl;
        model_info << "\t type: " << type << std::endl;
        model_info << "\t device: " << device << std::endl;
        _print_op_inputs(model_info, op);
        _print_op_outputs(model_info, graph, op);
        model_info << std::endl;
    }
    model_info.close();
}