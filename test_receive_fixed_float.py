# -*- coding: utf-8 -*-
import spidev
import time
import struct

# === CONFIGURAÇÃO DO SPI ===
spi = spidev.SpiDev()
spi.open(0, 0)  # <- USA ESTE! /dev/spidev0.0
spi.max_speed_hz = 10000000

# === Funções de controle do MCP2515 ===

def reset_mcp2515():
    spi.xfer2([0xC0])  # Comando de reset
    time.sleep(0.1)

def set_normal_mode():
    spi.xfer2([0x02, 0x0F, 0x00])  # Escreve 0x00 no registrador CANCTRL (modo normal)
    time.sleep(0.01)

def read_canctrl():
    resp = spi.xfer2([0x03, 0x0F, 0x00])  # Lê registrador CANCTRL
    return resp[2]

def check_rx_status():
    status = spi.xfer2([0xB0, 0x00])[1]  # Comando RX_STATUS
    return status & 0xC0  # Verifica se RX0/RX1 tem dado

def read_can_frame():
    response = spi.xfer2([0x90] + [0x00]*13)  # Comando READ RXB0
    sid_high = response[1]
    sid_low = response[2]
    can_id = (sid_high << 3) | (sid_low >> 5)
    dlc = response[5] & 0x0F
    data = response[6:6+dlc]
    return can_id, data

def bytes_to_float(data_bytes):
    return struct.unpack('<f', bytearray(data_bytes))[0]

# === Execução principal ===

if __name__ == "__main__":
    print("Inicializando MCP2515...")
    reset_mcp2515()
    set_normal_mode()
    mode = read_canctrl()
    print(f"🧪 CANCTRL = 0x{mode:02X} (esperado: 0x00)")

    if mode != 0x00:
        print("❌ ERRO: MCP2515 não está em modo normal. Verifica conexões e SPI.")
    else:
        print("✅ MCP2515 em modo normal.")
        print("🕓 Esperando dados CAN do Arduino (ID 0x100, valor: 3.14)...")

        try:
            while True:
                if check_rx_status():
                    can_id, data = read_can_frame()
                    print(f"📥 ID: 0x{can_id:X}, Data: {data}")
                    if can_id == 0x100 and len(data) >= 4:
                        val = bytes_to_float(data[:4])
                        print(f"🎯 Valor recebido: {val:.2f}")
                time.sleep(0.05)
        except KeyboardInterrupt:
            print("Encerrado.")
