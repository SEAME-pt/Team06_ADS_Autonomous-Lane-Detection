import spidev
import time
import struct

spi = spidev.SpiDev()
spi.open(0, 0)
spi.max_speed_hz = 1000000

def reset_mcp2515():
    spi.xfer2([0xC0])
    time.sleep(0.1)

def set_baudrate_500kbps_16MHz():
    spi.xfer2([0x02, 0x28, 0x86])  # CNF3
    spi.xfer2([0x02, 0x29, 0xF0])  # CNF2
    spi.xfer2([0x02, 0x2A, 0x00])  # CNF1

def set_normal_mode():
    spi.xfer2([0x02, 0x0F, 0x00])  # Modo normal
    time.sleep(0.01)

def check_rx_status():
    status = spi.xfer2([0xB0, 0x00])[1]
    return status & 0xC0

def read_can_message():
    resp = spi.xfer2([0x90] + [0x00]*13)
    sid_high = resp[1]
    sid_low = resp[2]
    can_id = (sid_high << 3) | (sid_low >> 5)
    dlc = resp[5] & 0x0F
    data = resp[6:6+dlc]
    spi.xfer2([0x02, 0x2C, 0x01])  # Limpa RX0IF
    return can_id, data

def bytes_to_float(b):
    return struct.unpack('<f', bytearray(b))[0]

if __name__ == "__main__":
    reset_mcp2515()
    set_baudrate_500kbps_16MHz()
    set_normal_mode()
    print("Pronto para receber mensagens CAN...")

    while True:
        if check_rx_status():
            can_id, data = read_can_message()
            print(f"ID: 0x{can_id:X} Dados: {list(data)}")
            if len(data) >= 4:
                val = bytes_to_float(data[:4])
                print(f"Valor float recebido: {val:.2f}")
        time.sleep(0.1)
