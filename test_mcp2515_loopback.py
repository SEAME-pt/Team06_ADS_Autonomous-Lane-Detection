import spidev
import time
import struct

spi = spidev.SpiDev()
spi.open(0, 0)  # /dev/spidev0.0, ajusta se necessário
spi.max_speed_hz = 1000000  # 1MHz é bom para MCP2515

def reset_mcp2515():
    spi.xfer2([0xC0])
    time.sleep(0.1)

def set_loopback_mode():
    # Modo loopback = 0x40 no registrador CANCTRL (endereço 0x0F)
    # Comando para escrever: 0x02 addr data
    spi.xfer2([0x02, 0x0F, 0x40])
    time.sleep(0.01)
    # Ler para confirmar
    resp = spi.xfer2([0x03, 0x0F, 0x00])
    return resp[2]

def send_can_message(can_id, data_bytes):
    # Iniciar envio de dados no buffer TXB0 (comando 0x40)
    # Definir ID padrão (11 bits) no TXB0SIDH/TXB0SIDL (endereços 0x31 e 0x32)
    sid_high = (can_id >> 3) & 0xFF
    sid_low = (can_id & 0x07) << 5
    
    # Enviar comando WRITE para TXB0, endereço 0x31 (TXB0SIDH)
    # O buffer precisa do ID alto, ID baixo, EID8, EID0, DLC, e depois os dados
    msg = [0x02, 0x31, sid_high, sid_low, 0x00, 0x00, len(data_bytes)] + list(data_bytes)
    spi.xfer2(msg)
    
    # Enviar comando RTS (Request To Send) para buffer 0 (0x81)
    spi.xfer2([0x81])

def check_rx_status():
    status = spi.xfer2([0xB0, 0x00])[1]
    # Os bits 6 e 7 indicam se há mensagem no RXB0 ou RXB1
    return status & 0xC0

def read_can_message():
    # Ler buffer RXB0 (comando 0x90)
    resp = spi.xfer2([0x90] + [0x00]*13)
    sid_high = resp[1]
    sid_low = resp[2]
    can_id = (sid_high << 3) | (sid_low >> 5)
    dlc = resp[5] & 0x0F
    data = resp[6:6+dlc]
    
    # Limpar o flag de RX0IF (bit 0 da INTFLAG - endereço 0x2C)
    spi.xfer2([0x02, 0x2C, 0x01])
    
    return can_id, data

def float_to_bytes(f):
    return struct.pack('<f', f)

def bytes_to_float(b):
    return struct.unpack('<f', bytearray(b))[0]

if __name__ == "__main__":
    print("Resetando MCP2515...")
    reset_mcp2515()
    
    print("Configurando MCP2515 para modo loopback...")
    mode = set_loopback_mode()
    if mode != 0x40:
        print(f"Erro: modo loopback não configurado corretamente (lido 0x{mode:02X})")
        exit(1)
    print("✅ MCP2515 em modo loopback.")
    
    test_value = 3.14
    print(f"Enviando valor float {test_value} no ID 0x100...")
    send_can_message(0x100, float_to_bytes(test_value))
    
    print("Aguardando retorno da mensagem no RX buffer...")
    timeout = time.time() + 2  # Timeout de 2 segundos
    while time.time() < timeout:
        if check_rx_status():
            can_id, data = read_can_message()
            print(f"Recebido ID: 0x{can_id:X} | Dados: {list(data)}")
            if can_id == 0x100 and len(data) >= 4:
                val = bytes_to_float(data[:4])
                print(f"✔️ Valor float decodificado: {val:.2f}")
            break
        time.sleep(0.05)
    else:
        print("⏰ Timeout: Nenhuma mensagem recebida no RX buffer.")
