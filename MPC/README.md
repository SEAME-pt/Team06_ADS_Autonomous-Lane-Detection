
## Compile process_mask
g++ -o process_mask process_mask.cpp `pkg-config --cflags --libs opencv4`
./process_mask

## Compile dynamic_model
g++ -o modelo_veiculo modelo_veiculo.cpp
./modelo_veiculo

## Compile MPC
g++ -o mpc mpc.cpp -I/usr/include/eigen3
./mpc

