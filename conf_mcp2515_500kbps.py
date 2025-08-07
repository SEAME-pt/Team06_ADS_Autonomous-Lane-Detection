# -*- coding: utf-8 -*-
import spidev
import time

spi = spidev.SpiDev()
spi.open(0, 0)
spi.max_speed_hz = 1000000

def reset_mcp2515():
    spi.xfer2([0xC0])  # RESET comando
    time.sleep(0.1)

def write_register(addr, value):
    spi.xfer2([0x02, addr, value])

def read_register(addr):
    resp = spi.xfer2([0x03, addr, 0x00])
    return resp[2]

def set_baudrate_500kbps_16MHz():
    write_register(0x28, 0x86)  # CNF3
    write_register(0x29, 0xF0)  # CNF2
    write_register(0x2A, 0x00)  # CNF1

def set_baudrate_500kbps_8MHz():
    write_register(0x28, 0x05)  # CNF3
    write_register(0x29, 0xB1)  # CNF2
    write_register(0x2A, 0x01)  # CNF1

def set_normal_mode():
    write_register(0x0F, 0x00)  # CANCTRL = 0x00 para modo normal
    time.sleep(0.01)
    return read_register(0x0F)

def detect_and_configure_baudrate():
    reset_mcp2515()
    
    # Tenta 16 MHz
    set_baudrate_500kbps_16MHz()
    mode = set_normal_mode()
    if mode == 0x00:
        print("Detectado cristal 16 MHz. MCP2515 configurado para 500 kbps em modo normal.")
        return "16MHz"
    
    # Se falhou, tenta 8 MHz
    reset_mcp2515()
    set_baudrate_500kbps_8MHz()
    mode = set_normal_mode()
    if mode == 0x00:
        print("Detectado cristal 8 MHz. MCP2515 configurado para 500 kbps em modo normal.")
        return "8MHz"
    
    print(f"Erro: Nao foi possivel configurar o MCP2515 para modo normal. CANCTRL = 0x{mode:02X}")
    return None

if __name__ == "__main__":
    cristal = detect_and_configure_baudrate()
