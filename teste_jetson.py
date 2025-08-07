import spidev
import time

# Comandos do MCP2515 (valores hexadecimais do datasheet)
RESET = 0xC0
READ = 0x03
WRITE = 0x02
READ_STATUS = 0xA0
READ_RX_BUFFER = 0x90 # Exemplo para RXB0

# Configuração do SPI
SPI_BUS = 0  # Pode ser 0 ou 1 no Jetson
SPI_DEVICE = 0 # Chip Select 0

# Inicializa o SPI
spi = spidev.SpiDev()
spi.open(SPI_BUS, SPI_DEVICE)
spi.max_speed_hz = 10000000 # 10 MHz
spi.mode = 0b00 # Modo 0 (CPOL=0, CPHA=0)

# Função para inicializar o MCP2515
def init_mcp2515():
    # Enviar o comando de reset
    spi.xfer2([RESET])
    time.sleep(0.1)
    
    # Exemplo: Escrever no registo de configuração
    # Isto é apenas um exemplo, os valores exatos dependem da sua configuração
    # Terá de consultar o datasheet do MCP2515 para os valores exatos de BRP, SJW, etc.
    # Exemplo: Configurar a taxa de bits para 500kbps
    # spi.xfer2([WRITE, 0x2A, 0x00, 0x00, 0x01]) # Exemplo de configuração

def read_can_message():
    # 1. Verificar se há uma mensagem nova (usando READ_STATUS)
    status = spi.xfer2([READ_STATUS])[0]
    
    # 2. Se houver, ler os dados do buffer de receção
    # Este é um processo complexo que envolve ler registos de ID e data
    # Exemplo (muito simplificado):
    if status & 0x01: # RXB0 tem mensagem
        message_data = spi.xfer2([READ_RX_BUFFER, 0, 0, 0, 0, 0, 0, 0, 0])
        # Aqui, terá de extrair o ID e os dados dos bytes recebidos
        # Depois, decodificar os 4 bytes para um float
        
        # ... Decodificação
        # speed_mps = struct.unpack('<f', message_data[5:9])
        
        # ... e depois limpar o buffer para a próxima mensagem
        # spi.xfer2([WRITE, CANINTF_REG, 0x00])

# --- Loop principal ---
init_mcp2515()
try:
    while True:
        read_can_message()
        time.sleep(0.01)

except KeyboardInterrupt:
    spi.close()