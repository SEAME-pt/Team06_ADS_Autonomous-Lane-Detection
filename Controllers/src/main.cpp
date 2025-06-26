#include "../inc/BackMotors.hpp"
#include "../inc/FServo.hpp"
#include "../inc/Controller.hpp"
#include "../inc/ControlAssembly.hpp"

int main()
{
    ControlAssembly controlAssembly;
    return 0;
}

/* int main() {
    // Conectar o manipulador de sinal ao SIGINT
    signal(SIGINT, signalHandler);

    try {
        FServo servo;
        BackMotors motors;

        // Inicializar servo e motores
        if (!servo.init_servo()) {
            std::cerr << "Falha na inicialização do servo.\n";
            return 1;
        }

        if (!motors.init_motors()) {
            std::cerr << "Falha na inicialização dos motores.\n";
            return 1;
        }

        std::cout << "Inicialização completa. Testando funções...\n";

        // Testar controle de direção
        std::cout << "Definindo ângulo de direção para -45 graus...\n";
        servo.set_steering(-45);
        std::this_thread::sleep_for(std::chrono::seconds(2));

        std::cout << "Definindo ângulo de direção para +45 graus...\n";
        servo.set_steering(45);
        std::this_thread::sleep_for(std::chrono::seconds(2));

        std::cout << "Centralizando direção...\n";
        servo.set_steering(0);
        std::this_thread::sleep_for(std::chrono::seconds(2));

        // Testar controle de velocidade
        //std::cout << "Acelerando para frente a 50% da velocidade...\n";
        //motors.setSpeed(50);
        //std::this_thread::sleep_for(std::chrono::seconds(3));

        //std::cout << "Revertendo para trás a 30% da velocidade...\n";
        //motors.setSpeed(-30);
        //std::this_thread::sleep_for(std::chrono::seconds(3));

        //std::cout << "Parando motores...\n";
        //motors.setSpeed(0);

        std::cout << "Teste concluído com sucesso.\n";
    } catch (const std::exception &e) {
        std::cerr << "Erro: " << e.what() << "\n";
        return 1;
    }

    return 0;
} */
