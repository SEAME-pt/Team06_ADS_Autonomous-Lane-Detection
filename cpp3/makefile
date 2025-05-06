TARGET = test
SRC = main.cpp infer.cpp controller.cpp

CXX = g++
CXXFLAGS = -O2 -std=c++17

INCLUDES = -I/usr/include/opencv4

LIBS = -lnvinfer -lnvonnxparser -lcudart \
       -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_videoio

CUDA_PATH = /usr/local/cuda
LIBS += -L$(CUDA_PATH)/lib64

all:
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET) $(INCLUDES) $(LIBS)

clean:
	rm -f $(TARGET)

.PHONY: all clean
