#include "tf_utils.h"

void print_tf_version()
{
    std::cout << "TensorFlow Version is : " << TF_Version() << std::endl;
}

void _print_op_inputs(std::ofstream &model_info, TF_Operation *op)
{
    //returns number of inputs for an operation
    const int num_inputs = TF_OperationNumInputs(op);

    model_info << "\t Number of inputs: " << num_inputs << std::endl;

    for (int i = 0; i < num_inputs; ++i)
    {
        const TF_Input input = {op, i};
        const TF_DataType type = TF_OperationInputType(input);
        model_info << "\t\t" << i << " type : " << _tf_data_type_to_string(type) << std::endl;
    }
}

void _print_op_outputs(std::ofstream &model_info, TF_Graph *graph, TF_Operation *op)
{
    TF_Status *status = TF_NewStatus();
    const int num_outputs = TF_OperationNumOutputs(op);

    model_info << "\t Number of outputs: " << num_outputs << std::endl;

    for (int i = 0; i < num_outputs; ++i)
    {
        const TF_Output output = {op, i};
        const TF_DataType type = TF_OperationOutputType(output);
        model_info << "\t\t" << i << " type : " << _tf_data_type_to_string(type);

        const int num_dims = TF_GraphGetTensorNumDims(graph, output, status);

        if (TF_GetCode(status) != TF_OK)
        {
            model_info << "Can't get tensor dimensionality" << std::endl;
            continue;
        }

        if (num_dims == -1)
        {
            TF_GraphGetTensorShape(graph, output, nullptr, num_dims, status);
        }
        else
        {
            std::vector<int64_t> dims(num_dims);
            TF_GraphGetTensorShape(graph, output, dims.data(), num_dims, status);

            //error handling
            if (TF_GetCode(status) != TF_OK)
            {
                model_info << "Can't get get tensor shape" << std::endl;
                continue;
            }

            model_info << "      dims: " << num_dims << " [";
            for (int j = 0; j < num_dims; ++j)
            {
                model_info << dims[j];
                if (j < num_dims - 1)
                {
                    model_info << ",";
                }
            }
            model_info << "]" << std::endl;
        }
    }

    TF_DeleteStatus(status);
}

void _print_tn_info(TF_Graph *graph, const char *layer_name)
{
    std::string file_name(layer_name);
    file_name += ".txt";
    std::ofstream info;
    info.open(file_name);

    info << "Tensor: " << layer_name;
    TF_Operation *op = TF_GraphOperationByName(graph, layer_name);

    if (op == nullptr)
    {
        info << "Could not get " << layer_name << std::endl;
        return;
    }

    const int num_inputs = TF_OperationNumInputs(op);
    const int num_outputs = TF_OperationNumOutputs(op);
    info << " inputs: " << num_inputs << " outputs: " << num_outputs << std::endl;

    _print_op_inputs(info, op);

    _print_op_outputs(info, graph, op);

    info.close();
}
const char *_tf_data_type_to_string(TF_DataType data_type)
{
    switch (data_type)
    {
    case TF_FLOAT:
        return "TF_FLOAT";
    case TF_DOUBLE:
        return "TF_DOUBLE";
    case TF_INT32:
        return "TF_INT32";
    case TF_UINT8:
        return "TF_UINT8";
    case TF_INT16:
        return "TF_INT16";
    case TF_INT8:
        return "TF_INT8";
    case TF_STRING:
        return "TF_STRING";
    case TF_COMPLEX64:
        return "TF_COMPLEX64";
    case TF_INT64:
        return "TF_INT64";
    case TF_BOOL:
        return "TF_BOOL";
    case TF_QINT8:
        return "TF_QINT8";
    case TF_QUINT8:
        return "TF_QUINT8";
    case TF_QINT32:
        return "TF_QINT32";
    case TF_BFLOAT16:
        return "TF_BFLOAT16";
    case TF_QINT16:
        return "TF_QINT16";
    case TF_QUINT16:
        return "TF_QUINT16";
    case TF_UINT16:
        return "TF_UINT16";
    case TF_COMPLEX128:
        return "TF_COMPLEX128";
    case TF_HALF:
        return "TF_HALF";
    case TF_RESOURCE:
        return "TF_RESOURCE";
    case TF_VARIANT:
        return "TF_VARIANT";
    case TF_UINT32:
        return "TF_UINT32";
    case TF_UINT64:
        return "TF_UINT64";
    default:
        return "Unknown";
    }
}
void tf_freeBuffer(void *data, size_t length)
{
    free(data);
}

void tf_deallocator(void *ptr, size_t len, void *arg)
{
    free((void *)ptr);
}