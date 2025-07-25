#include "ZmqPublisher.hpp"
#include <iostream>
#include <thread>
#include <chrono>
#include <json/json.h> // Para usar Json::Value

int main() {
    // Definir o host e a porta onde o publicador fará o BIND.
    // Isso deve corresponder ao endereço que o ZmqSubscriber do seu colega está conectando.
    const std::string PUBLISHER_HOST = "127.0.0.1";
    const int PUBLISHER_PORT = 5558;

    ZmqPublisher* publisher = nullptr; // Ponteiro para a instância do publicador

    try {
        // Cria uma instância do publicador
        publisher = new ZmqPublisher(PUBLISHER_HOST, PUBLISHER_PORT);

        // Verifica se o publicador conseguiu ligar (bind)
        if (!publisher->isConnected()) {
            std::cerr << "Nao foi possivel iniciar o ZMQ Publisher. Saindo." << std::endl;
            return 1;
        }

        // Exemplo de dados para enviar (usando Json::Value)
        Json::Value data1;
        data1["tipo"] = "sensor_temperatura";
        data1["valor"] = 25.7;
        data1["unidade"] = "celsius";
        data1["timestamp"] = std::chrono::duration_cast<std::chrono::milliseconds>(
                                 std::chrono::system_clock::now().time_since_epoch()
                             ).count(); // Timestamp em ms

        // Publica a primeira mensagem
        if (publisher->publishMessage(data1)) {
            std::cout << "Mensagem 1 publicada com sucesso." << std::endl;
        } else {
            std::cerr << "Falha ao publicar mensagem 1." << std::endl;
        }

        std::this_thread::sleep_for(std::chrono::seconds(1)); // Espera 1 segundo

        Json::Value data2;
        data2["tipo"] = "status_componente";
        data2["componente_id"] = "ABC-123";
        data2["status"] = "OK";
        data2["last_check"] = std::chrono::duration_cast<std::chrono::milliseconds>(
                                 std::chrono::system_clock::now().time_since_epoch()
                             ).count();

        // Publica a segunda mensagem
        if (publisher->publishMessage(data2)) {
            std::cout << "Mensagem 2 publicada com sucesso." << std::endl;
        } else {
            std::cerr << "Falha ao publicar mensagem 2." << std::endl;
        }

        std::this_thread::sleep_for(std::chrono::seconds(1));

        Json::Value data3;
        data3["alerta"] = true;
        data3["severidade"] = "alta";
        data3["descricao"] = "Nivel critico de energia.";

        if (publisher->publishMessage(data3)) {
            std::cout << "Mensagem 3 publicada com sucesso." << std::endl;
        } else {
            std::cerr << "Falha ao publicar mensagem 3." << std::endl;
        }


    } catch (const std::exception& e) {
        std::cerr << "Erro fatal no programa principal: " << e.what() << std::endl;
    }

    // Libera a memória alocada para o publicador
    if (publisher) {
        delete publisher;
    }

    // O contexto ZMQ é uma instância estática (`zmq::context_t::instance()`)
    // e será terminado automaticamente ao final do programa.
    // Se você tivesse criado um contexto explicitamente (e não como instância global),
    // precisaria terminá-lo aqui (e.g., `context.term();`).

    std::cout << "Programa finalizado." << std::endl;
    return 0;
}