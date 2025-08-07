import spidev
import time
import struct

# Inicializa SPI
spi = spidev.SpiDev()
spi.open(0, 0)  # SPI0.0 (ajuste se necessario)
spi.max_speed_hz = 10000000

# Reseta o MCP2515
def reset_mcp2515():
    spi.xfer2([0xC0])  # RESET
    time.sleep(0.1)

# Modo normal de operacao
def set_normal_mode():
    spi.xfer2([0x02, 0x0F, 0x00])  # Write 0x00 to CANCTRL
    time.sleep(0.01)

# Verifica se ha dados recebidos
def check_rx_status():
    status = spi.xfer2([0xB0, 0x00])[1]
    return status & 0xC0  # 0x40 (RXB0 full), 0x80 (RXB1 full)

# Le uma mensagem CAN do RXB0
def read_can_frame():
    response = spi.xfer2([0x90] + [0x00]*13)  # Read RXB0
    sid_high = response[1]
    sid_low = response[2]
    can_id = (sid_high << 3) | (sid_low >> 5)
    dlc = response[5] & 0x0F
    data = response[6:6+dlc]
    return can_id, data

# Converte 4 bytes para float
def bytes_to_float(data_bytes):
    return struct.unpack('<f', bytearray(data_bytes))[0]

# Programa principal
if __name__ == "__main__":
    print("Inicializando MCP2515...")
    reset_mcp2515()
    set_normal_mode()

    print("Aguardando velocidade do Arduino via CAN...\nPressione Ctrl+C para sair.\n")
    try:
        while True:
            if check_rx_status():
                can_id, data = read_can_frame()
                print(f"Recebido ID: 0x{can_id:X} | Dados brutos: {data}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nEncerrado.")
