#include <iostream>
#include <string>
#include <zmq.hpp> // Inclui a biblioteca ZeroMQ C++
#include <thread>  // Para std::this_thread::sleep_for
#include <chrono>  // Para std::chrono::milliseconds

int main() {
    // Crie o contexto ZeroMQ. Um contexto é necessário para qualquer operação ZeroMQ.
    // O argumento é o número de threads de I/O a serem usadas.
    zmq::context_t context(1);

    // Crie um socket do tipo SUB (Subscriber)
    zmq::socket_t subscriber(context, zmq::socket_type::sub);

    // O endereço do seu Publisher.
    // DEVE SER O MESMO ENDEREÇO ONDE O SEU ZmqPublisher ESTÁ FAZENDO BIND.
    const std::string publisher_address = "tcp://127.0.0.1:5558";

    std::cout << "ZMQ Subscriber: Tentando conectar a " << publisher_address << "..." << std::endl;

    try {
        // Conecte-se ao Publisher
        // Note que o Subscriber FAZ CONNECT, enquanto o Publisher FAZ BIND.
        subscriber.connect(publisher_address);

        // Assine todas as mensagens (string vazia significa 'tudo')
        // Em um cenário real, você poderia ter tópicos específicos aqui (ex: "telemetria")
        subscriber.set(zmq::sockopt::subscribe, "");

        // Opcional: Definir um alto watermark de recebimento para evitar perda de mensagens
        // se o processamento for lento.
        int rcvhwm = 1000;
        subscriber.set(zmq::sockopt::rcvhwm, rcvhwm);

        std::cout << "ZMQ Subscriber: Conectado. Aguardando mensagens do Publisher..." << std::endl;

        while (true) {
            zmq::message_t received_message;
            zmq::recv_result_t result;

            try {
                // Recebe a mensagem. ZMQ_DONTWAIT tornaria a chamada não bloqueante.
                // Aqui, a chamada é bloqueante, esperando por uma mensagem.
                result = subscriber.recv(received_message, zmq::recv_flags::none);

                if (result) {
                    // Converte a mensagem ZeroMQ para uma string C++
                    std::string message_str(static_cast<char*>(received_message.data()), received_message.size());
                    std::cout << "ZMQ Subscriber: Mensagem Recebida: " << message_str << std::endl;
                } else {
                    // result pode ser nullptr se houver um erro ou se o socket for fechado
                    std::cerr << "ZMQ Subscriber ERROR: Falha ao receber mensagem." << std::endl;
                    // Pequena pausa para evitar loop infinito em caso de erro persistente
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                }
            } catch (const zmq::error_t& e) {
                std::cerr << "ZMQ Subscriber ERROR ao receber: " << e.what() << std::endl;
                break; // Sair do loop em caso de erro crítico de ZMQ
            } catch (const std::exception& e) {
                std::cerr << "ZMQ Subscriber ERROR inesperado ao receber: " << e.what() << std::endl;
                break; // Sair do loop em caso de erro inesperado
            }
        }

    } catch (const zmq::error_t& e) {
        std::cerr << "ZMQ Subscriber ERROR: Nao foi possivel conectar: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "ZMQ Subscriber ERROR: Erro inesperado: " << e.what() << std::endl;
    }

    std::cout << "ZMQ Subscriber: Encerrado." << std::endl;

    // O contexto será destruído automaticamente ao sair da função main,
    // o que também encerrará o socket.
    return 0;
}