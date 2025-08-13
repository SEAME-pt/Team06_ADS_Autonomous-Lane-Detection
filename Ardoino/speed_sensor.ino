#include <SPI.h>
#include "mcp2515_can.h"

// Definições do CAN-Bus Shield
#define CAN_2515
#define MCP_16MHZ    0
#define MCP_8MHZ     1
const int SPI_CS_PIN = 9;
const int CAN_INT_PIN = 2;
mcp2515_can CAN(SPI_CS_PIN);

// Definições do Sensor de Velocidade
const int ENCODER_PIN = 3;
const unsigned int pulsesPerRevolution = 18; // 18 furos no disco
const float wheelDiameter_mm = 67.0; // Diâmetro da roda em milímetros
const float wheelCircumference_m = (wheelDiameter_mm / 1000.0) * PI; // Circunferência em metros
const unsigned long measurementInterval = 500; // Intervalo de medição em milissegundos (0.5s)

volatile unsigned long pulseCount = 0;
unsigned long lastMeasurementTime = 0;
float currentSpeed_mps = 0.0; // m/s

// Interrupção para contar os pulsos
void pulseISR() {
    pulseCount++;
}

void setup() {
    Serial.begin(9600);
    
    // Configura o pino do encoder como entrada com pull-up interno
    pinMode(ENCODER_PIN, INPUT_PULLUP);
    
    // Ativa a interrupção no pino do encoder
    attachInterrupt(digitalPinToInterrupt(ENCODER_PIN), pulseISR, RISING);

    // Inicializa o CAN-Bus Shield
    while (CAN.begin(CAN_500KBPS, MCP_8MHZ) != CAN_OK) {
        Serial.println("CAN BUS Initialization Failed! Retrying...");
        delay(1000);
    }
    Serial.println("CAN BUS Initialized successfully!");
}

void loop() {
    // Verifica se já passou o tempo de medição
    if (millis() - lastMeasurementTime >= measurementInterval) {
        // Calcula a quantidade de pulsos no intervalo de tempo
        unsigned long pulsesInInterval = pulseCount;
        
        // Zera o contador para a próxima medição
        pulseCount = 0;
        
        // Calcula a velocidade
        float revolutions = (float)pulsesInInterval / pulsesPerRevolution;
        currentSpeed_mps = (revolutions * wheelCircumference_m) / (measurementInterval / 1000.0);
        
        // Atualiza o tempo da última medição
        lastMeasurementTime = millis();
        
        // Envia a velocidade via CAN-Bus
        sendSpeedViaCAN(currentSpeed_mps);
        
        // Exibe a velocidade no Monitor Serial (para depuração)
        Serial.print("Velocidade: ");
        Serial.print(currentSpeed_mps, 2); // 2 casas decimais
        Serial.println(" m/s");
    }
}

// Função para enviar dados pela rede CAN
void sendSpeedViaCAN(float speed) {
    long id = 0x100; // ID do pacote CAN (pode ser ajustado)
    byte data[8];
    
    // Converte o float em 4 bytes para o pacote CAN
    memcpy(data, &speed, sizeof(speed));
    
    // Envia o pacote
    CAN.sendMsgBuf(id, 0, sizeof(speed), data);
}