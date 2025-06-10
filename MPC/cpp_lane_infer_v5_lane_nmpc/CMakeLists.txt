cmake_minimum_required(VERSION 3.10)
project(LaneDetection)

# Definir o padrão C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Encontrar pacotes necessários
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

# Configurar CasADi manualmente
set(CASADI_INCLUDE_DIRS /usr/local/include)
set(CASADI_LIBRARIES /usr/local/lib/libcasadi.so)
if(NOT EXISTS "${CASADI_INCLUDE_DIRS}/casadi/casadi.hpp")
    message(FATAL_ERROR "CasADi include não encontrado em ${CASADI_INCLUDE_DIRS}/casadi/casadi.hpp")
endif()
if(NOT EXISTS "${CASADI_LIBRARIES}")
    message(FATAL_ERROR "CasADi library não encontrada em ${CASADI_LIBRARIES}")
endif()
message(STATUS "CasADi configurado manualmente!")
message(STATUS "CASADI_INCLUDE_DIRS: ${CASADI_INCLUDE_DIRS}")
message(STATUS "CASADI_LIBRARIES: ${CASADI_LIBRARIES}")

# Definir caminhos para o TensorRT
set(TensorRT_ROOT "/usr/src/tensorrt")
find_library(TensorRT_LIB nvinfer HINTS ${TensorRT_ROOT}/lib)
find_library(NvInferPlugin_LIB nvinfer_plugin HINTS ${TensorRT_ROOT}/lib)
find_library(NvOnnxParser_LIB nvonnxparser HINTS ${TensorRT_ROOT}/lib)

# Incluir diretórios
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${TensorRT_ROOT}/include
    ${CMAKE_SOURCE_DIR}
    ${CASADI_INCLUDE_DIRS}
)

# Adicionar executável
add_executable(lane_detection
    main.cpp
    lane_detection.cpp
    mask_processor.cpp
    nmpc_controller.cpp
)

# Linkar bibliotecas
target_link_libraries(lane_detection
    ${OpenCV_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${TensorRT_LIB}
    ${NvInferPlugin_LIB}
    ${NvOnnxParser_LIB}
    ${CASADI_LIBRARIES}
    rt
)

# Configurações adicionais para Jetson
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Ofast")

# Mensagens de depuração
message(STATUS "Final CASADI_INCLUDE_DIRS: ${CASADI_INCLUDE_DIRS}")
message(STATUS "Final CASADI_LIBRARIES: ${CASADI_LIBRARIES}")