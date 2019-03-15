CC      = c++
CXXFLAGS +=-std=c++11
LDFLAGS = -I/usr/local/include \
-I/project_dir/include\
-I/3rdparty/opencv340/release/include \
-I/3rdparty/tensorflow_c_api/include/tensorflow/c \
-L/3rdparty/opencv340/release/lib \
-L/3rdparty/tensorflow_c_api/lib  \
-L/usr/local/lib \
-ltensorflow -lopencv_core -lopencv_aruco -lopencv_bgsegm -lopencv_calib3d -lopencv_ccalib -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_video -lopencv_videoio

all: segmentation

segmentation: src/main.cc src/semantic.cc src/tf_model.cc src/tf_utils.cc include/semantic.h include/tf_model.h include/tf_utils.h

	$(CC) $(CXXFLAGS) -o $@ $^ $(LDFLAGS) 

.PHONY: clean

clean:
	rm *.txt
