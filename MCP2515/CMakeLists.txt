cmake_minimum_required(VERSION 3.10)
project(canbus_app)

set(CMAKE_CXX_STANDARD 17)

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
    MCP2515Constants.hpp # Adicionado o novo ficheiro de constantes
)

# Linkar as bibliotecas necessárias
# A thread é necessária para a CanBusManager
target_link_libraries(canbus_app PRIVATE pthread)