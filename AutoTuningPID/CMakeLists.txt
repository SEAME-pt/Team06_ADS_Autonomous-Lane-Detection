cmake_minimum_required(VERSION 3.10)
project(canbus_app)

# Define o padrão C++ para C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Inclui os diretórios para encontrar os cabeçalhos
include_directories(${CMAKE_SOURCE_DIR})

# Adiciona todos os ficheiros fonte
add_executable(canbus_app
    main.cpp
    CanBusManager.cpp
    MCP2515Controller.cpp
    MCP2515Configurator.cpp
    SPIController.cpp
    CANMessageProcessor.cpp
    SpeedPIDController.cpp
    SpeedPIDTuner.cpp
    MCP2515Constants.hpp
    SpeedPIDController.hpp
    SpeedPIDTuner.hpp
)

# Linkar as bibliotecas necessárias
# pthread é necessário para CanBusManager (threads)
# Adicione libzmq se ZeroMQ for usado (opcional, descomente se necessário)
target_link_libraries(canbus_app PRIVATE pthread)
# target_link_libraries(canbus_app PRIVATE zmq)