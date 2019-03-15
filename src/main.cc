#include "semantic.h"
#include "opencv2/opencv.hpp"
#include <time.h>


using namespace cv;


int main()
{
	bool PROFILE = true;
	
        // your pb file address, input tesnor name, and output tensor names (can be multiple) should be provided here
	std::string pb_model_file = ".....pb";
	std::string input_tensor_name = "input tensor name";
	std::vector<std::string> output_tensor_names = {"output tensor name0","output tensor name1"};

	std::string filename = "vidfile.mp4";
        // 
	VideoCapture capture(filename);
	int frame_Height = capture.get(CAP_PROP_FRAME_HEIGHT);
	int frame_Width = capture.get(CAP_PROP_FRAME_WIDTH);
	std::cout << "h = " << frame_Height << ", w = " << frame_Width << std::endl;
	Mat frame;
        // instance of the task which uses the tensorflow model
	Semantic_Seg segmenter(frame_Height, frame_Width, 3, pb_model_file, input_tensor_name, output_tensor_names);
	// segmenter.model->inspect("z.txt"); // if you need to inspect the model prototxt like file will be generated 

	int frame_count = 0;
	bool cap_flag;
	capture.read(frame);
	if (!capture.isOpened())
		throw "Error when reading video file";

	Mat flat = frame.reshape(1);
	unsigned char *image = flat.data;
	Mat mask(frame_Height, frame_Width, CV_8UC1,cv::Scalar(0));

	int nframe = 0;
	clock_t t_net = 0, tic, tic0;
	if (PROFILE)
		tic = clock();
	while (true)
	{
		if (PROFILE)
			tic0 = clock();
		segmenter.get_mask(image, mask.data);
		if (PROFILE)
			t_net += (clock() - tic0);
		cap_flag = capture.read(frame);
		if (!cap_flag)
			break;
		flat = frame.reshape(1);
		
		if (PROFILE)
			std::cout << ++frame_count << ", fps = " << frame_count * CLOCKS_PER_SEC / ((float)(t_net)) << std::endl;
		image = flat.data;
		// cv::imshow("segmented...", mask);
		// char key = cv::waitKey(1);
		// if (key == 'q')
			break;
	}
	if (PROFILE)
        {
		std::cout << "Number of frames = " << frame_count
			  << " : fps = " << frame_count * CLOCKS_PER_SEC / ((float)(clock()- tic)) << std::endl;
        }
	return 0;
}
