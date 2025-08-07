import can

# Configura a interface CAN (vamos verificar o nome correto abaixo)
bus = can.interface.Bus(channel='can0', bustype='socketcan', bitrate=500000)

print("Esperando mensagens do Arduino...")
while True:
    msg = bus.recv(timeout=1.0)  # Espera 1 segundo por mensagem
    if msg:
        print(f"Mensagem recebida! ID: 0x{msg.arbitration_id:X}, Dados: {msg.data.hex()}")
    else:
        print("Nenhuma mensagem recebida...")