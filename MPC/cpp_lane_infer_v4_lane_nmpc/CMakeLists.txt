cmake_minimum_required(VERSION 3.10)
project(LaneDetection)

# Definir o padrão C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Encontrar pacotes necessários
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

# Definir caminhos para o TensorRT (ajuste conforme necessário)
set(TensorRT_ROOT "/usr/src/tensorrt")
find_library(TensorRT_LIB nvinfer HINTS ${TensorRT_ROOT}/lib)
find_library(NvInferPlugin_LIB nvinfer_plugin HINTS ${TensorRT_ROOT}/lib)

# Incluir diretórios
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${TensorRT_ROOT}/include
    ${CMAKE_SOURCE_DIR}
)

# Linkar diretórios de bibliotecas
link_directories(
    ${TensorRT_ROOT}/lib
)

# Adicionar executável
add_executable(lane_detection
    main.cpp
    lane_detection.cpp
    mask_processor.cpp
)

add_executable(nmpc nmpc.cpp)
target_link_libraries(nmpc ${OpenCV_LIBS})

# Linkar bibliotecas
target_link_libraries(lane_detection
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    ${TensorRT_LIB}
    ${NvInferPlugin_LIB}
)

# Configurações adicionais para Jetson
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Ofast")