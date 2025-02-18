import smbus

bus = smbus.SMBus(1)

# Endereço I2C da câmera
camera_address = 0x70

register = 0x00  

try:
    data = bus.read_byte_data(camera_address, register)
    print(f"Dado lido do registrador 0x{register:02X}: {data}")
except Exception as e:
    print(f"Erro ao ler dados do registrador 0x{register:02X}: {e}")
