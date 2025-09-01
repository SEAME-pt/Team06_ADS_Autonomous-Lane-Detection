import zmq
import time

# Código Python para testar receção numa porta PUB-SUB ZMQ na porta 5555
# O seu código C++ está a fazer SUB na porta 5555, este script faz PUB nessa porta

context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://*:5555")

print("Publisher ZMQ iniciado na porta 5555")

try:
    for i in range(100):
        msg = f"speed:{0.5 + i*0.1:.2f};"
        print(f"A enviar: {msg}")
        socket.send_string(msg)
        time.sleep(1)  # Aguarda 1 segundo entre mensagens
except KeyboardInterrupt:
    print("Parando publisher")
finally:
    socket.close()
    context.term()
