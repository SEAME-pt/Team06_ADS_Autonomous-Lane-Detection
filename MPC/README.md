

## dependencies
sudo apt-get update
sudo apt-get install libopencv-dev

## Compile process_mask
g++ -o process_mask process_mask.cpp `pkg-config --cflags --libs opencv4`
./process_mask

## Compile dynamic_model
g++ -o dynamic_model dynamic_model.cpp
./dynamic_model

## Compile MPC
g++ -o mpc mpc.cpp -I/usr/include/eigen3
./mpc

## Compile mpc_integrated
g++ -o mpc_integrated mpc_integrated.cpp `pkg-config --cflags --libs opencv4` -I/usr/include/eigen3
./mpc_integrated
