import spidev
import time

# Configuração da interface SPI
spi = spidev.SpiDev()
spi.open(0, 0)  # Bus 0, Device 0 (ajuste conforme necessário)
spi.max_speed_hz = 10000000

# Função para enviar um comando ao MCP2515
def send_command(cmd):
    spi.xfer2([cmd])

# Função para resetar o MCP2515
def reset_mcp2515():
    send_command(0xC0)  # RESET command
    time.sleep(0.1)

# Exemplo básico: enviar frame CAN (dados fictícios)
def send_frame():
    # Modo configuração
    spi.xfer2([0x02, 0x0F, 0x80])  # Escreve no registro CANCTRL para modo config
    time.sleep(0.01)

    # Configura bitrate (opcional aqui, normalmente feito uma vez)
    # Exemplo: Configura registradores CNF1/2/3 para 500kbps

    # Volta para modo normal
    spi.xfer2([0x02, 0x0F, 0x00])  # Modo normal

    # Escreve um frame de dados para TXB0
    # Exemplo: enviar 0x11223344 com ID 0x123
    spi.xfer2([
        0x40,       # Load TX buffer 0
        0x00,       # SIDH
        0x00,       # SIDL
        0x00,       # EID8
        0x00,       # EID0
        0x08,       # DLC = 8 bytes
        0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88  # Data bytes
    ])

    # Solicita envio
    spi.xfer2([0x81])  # RTS TXB0

    print("Frame CAN enviado!")

# Execução
if __name__ == "__main__":
    print("Resetando MCP2515...")
    reset_mcp2515()

    print("Enviando frame CAN...")
    send_frame()

    print("Finalizado.")
