import spidev
import time
import struct

spi = spidev.SpiDev()
spi.open(0, 0)
spi.max_speed_hz = 1000000

def reset_mcp2515():
    spi.xfer2([0xC0])
    time.sleep(0.1)

def set_loopback_mode():
    # Configura o MCP2515 em modo loopback (modo 0x40 no CANCTRL)
    spi.xfer2([0x02, 0x0F, 0x40])
    time.sleep(0.01)

def read_canctrl():
    resp = spi.xfer2([0x03, 0x0F, 0x00])
    return resp[2]

def send_can_message(can_id, data_bytes):
    # Escreve o ID e os dados no buffer TXB0
    sid_high = (can_id >> 3) & 0xFF
    sid_low = (can_id << 5) & 0xE0
    tx_buffer = [
        0x40,  # Load TX Buffer 0 command
        sid_high,
        sid_low,
        0x00,  # EID8
        0x00,  # EID0
        len(data_bytes),  # DLC
    ] + data_bytes
    spi.xfer2(tx_buffer)
    # Request to send buffer 0
    spi.xfer2([0x81])

def check_rx_status():
    status = spi.xfer2([0xB0, 0x00])[1]
    return status & 0xC0

def read_can_frame():
    response = spi.xfer2([0x90] + [0x00]*13)  # Read RXB0
    sid_high = response[1]
    sid_low = response[2]
    can_id = (sid_high << 3) | (sid_low >> 5)
    dlc = response[5] & 0x0F
    data = response[6:6+dlc]
    return can_id, data

def float_to_bytes(value):
    return list(struct.pack('<f', value))

def bytes_to_float(data_bytes):
    return struct.unpack('<f', bytearray(data_bytes))[0]

if __name__ == "__main__":
    print("Resetando MCP2515 e configurando modo loopback...")
    reset_mcp2515()
    set_loopback_mode()
    mode = read_canctrl()
    print(f"Modo CANCTRL = 0x{mode:02X} (esperado: 0x40)")

    if mode != 0x40:
        print("Erro: MCP2515 não entrou em modo loopback.")
    else:
        print("MCP2515 em modo loopback. Enviando valor 3.14 no ID 0x100...")
        val_float = 3.14
        data_bytes = float_to_bytes(val_float)
        send_can_message(0x100, data_bytes)

        print("Aguardando retorno do frame...")

        try:
            while True:
                if check_rx_status():
                    can_id, data = read_can_frame()
                    print(f"Recebido ID: 0x{can_id:X}, Data: {data}")
                    if can_id == 0x100 and len(data) >= 4:
                        val = bytes_to_float(data[:4])
                        print(f"Valor float decodificado: {val:.2f}")
                    break
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("Encerrado.")
