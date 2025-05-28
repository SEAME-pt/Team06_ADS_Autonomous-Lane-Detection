import carla
import cv2
import numpy as np
import time
import random
import sys
import os
import datetime
import math

# Configurações de exibição
WIDTH = 1024  # Largura da imagem capturada pelas câmeras
HEIGHT = 720  # Altura da imagem capturada pelas câmeras

# Variáveis globais para o dataset
frame_count = 0  # Contador de frames salvos
current_image = None  # Imagem RGB atual capturada
current_mask = None  # Máscara de segmentação atual
direction_forward = True  # Direção do trajeto (True: para frente, False: para trás)

def get_road_waypoints(world, vehicle):
    """Gera uma lista de waypoints que formam um segmento completo de estrada a partir da posição do veículo."""
    map = world.get_map()  # Obtém o mapa do mundo CARLA
    vehicle_location = vehicle.get_location()  # Obtém a localização atual do veículo
    current_waypoint = map.get_waypoint(vehicle_location)  # Encontra o waypoint mais próximo da posição do veículo
    waypoints = [current_waypoint]  # Inicializa a lista de waypoints com o waypoint atual
    
    # Coleta waypoints à frente
    next_waypoint = current_waypoint  # Começa pelo waypoint atual
    for i in range(200):  # Tenta coletar até 200 waypoints
        next_waypoints = next_waypoint.next(5.0)  # Obtém o próximo waypoint a 5 metros de distância
        if not next_waypoints:  # Se não houver mais waypoints, para o loop
            break
        next_waypoint = next_waypoints[0]  # Seleciona o primeiro waypoint da lista de próximos
        waypoints.append(next_waypoint)  # Adiciona o waypoint à lista
    
    print(f"Collected {len(waypoints)} waypoints")  # Imprime o número de waypoints coletados
    return waypoints  # Retorna a lista de waypoints

def drive_waypoints(world, vehicle, waypoints, direction_forward=True, speed_factor=0.5):
    """
    Controla o veículo para seguir os waypoints, com captura manual de imagens pelo usuário.
    Parâmetros:
    - world: Objeto do mundo CARLA
    - vehicle: Ator do veículo a ser controlado
    - waypoints: Lista de waypoints a seguir
    - direction_forward: Se False, segue a rota ao contrário
    - speed_factor: Controla a velocidade do veículo (0.1 a 1.0)
    Retorna:
    - True se a rota for concluída, False se o usuário interromper
    """
    global current_image , frame_count 
    frame_count = 0  # Reseta o contador de frames salvos
    
    if not direction_forward:  # Se a direção for inversa
        waypoints = waypoints[::-1]  # Inverte a lista de waypoints
    
    current_waypoint_index = 0  # Inicializa o índice do waypoint atual
    total_waypoints = len(waypoints)  # Calcula o número total de waypoints
    vehicle_control = carla.VehicleControl()  # Cria um objeto de controle do veículo
    
    running = True  # Flag para manter o loop principal ativo
    paused = False  # Flag para pausar a navegação
    
    print("\n=== NAVIGATION STARTED ===")  # Imprime mensagem de início
    print(f"Direction: {'FORWARD' if direction_forward else 'REVERSE'}")  # Indica a direção atual
    print("ENTER: Capture frame | P: Pause | +/-: Speed | ESC: Exit")  # Exibe comandos disponíveis
    
    while running and current_waypoint_index < total_waypoints:  # Loop enquanto ativo e waypoints não concluídos
        world.tick()  # Atualiza o mundo CARLA em modo síncrono
        
        target_waypoint = waypoints[current_waypoint_index]  # Obtém o waypoint alvo atual
        
        if not paused:  # Se a navegação não estiver pausada
            vehicle_transform = vehicle.get_transform()  # Obtém a transformação (posição/rotação) do veículo
            vehicle_location = vehicle_transform.location  # Obtém a localização do veículo
            target_location = target_waypoint.transform.location  # Obtém a localização do waypoint alvo
            
            direction = target_location - vehicle_location  # Calcula o vetor direção (waypoint - veículo)
            distance = math.sqrt(direction.x**2 + direction.y**2)  # Calcula a distância 2D até o waypoint
            
            if distance < 2.0:  # Se o veículo está a menos de 2 metros do waypoint
                current_waypoint_index += 1  # Avança para o próximo waypoint
                if current_waypoint_index % 10 == 0:  # A cada 10 waypoints
                    print(f"Waypoint {current_waypoint_index}/{total_waypoints}")  # Imprime progresso
                if current_waypoint_index >= total_waypoints:  # Se todos os waypoints foram percorridos
                    print("Route completed!")  # Imprime mensagem de conclusão
                    break
                continue  # Pula para a próxima iteração
            
            vehicle_forward = vehicle_transform.get_forward_vector()  # Obtém o vetor de direção do veículo
            dot = vehicle_forward.x * direction.x + vehicle_forward.y * direction.y  # Calcula o produto escalar
            cross = vehicle_forward.x * direction.y - vehicle_forward.y * direction.x  # Calcula o produto vetorial
            angle = math.atan2(cross, dot)  # Calcula o ângulo entre o veículo e o waypoint
            
            steering = max(-1.0, min(1.0, angle * 2.0))  # Converte o ângulo para valor de direção [-1, 1]
            
            vehicle_control.throttle = speed_factor  # Define a aceleração com base no speed_factor
            vehicle_control.steer = steering  # Define a direção do volante
            vehicle_control.brake = 0.0  # Remove o freio
            vehicle.apply_control(vehicle_control)  # Aplica o controle ao veículo
        
        if current_image is not None:  # Se há uma imagem RGB disponível
            img_display = current_image.copy()  # Cria uma cópia da imagem para exibição
            
            velocity = vehicle.get_velocity()  # Obtém a velocidade do veículo
            speed = 3.6 * np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)  # Calcula a velocidade em km/h
            current_steering = vehicle_control.steer  # Obtém o valor atual de direção
            
            # Adiciona informações à imagem exibida
            cv2.putText(img_display, f"Speed: {speed:.1f} km/h", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)  # Exibe velocidade
            cv2.putText(img_display, f"Steering: {current_steering:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)  # Exibe direção
            cv2.putText(img_display, f"Waypoint: {current_waypoint_index}/{total_waypoints}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)  # Exibe progresso de waypoints
            cv2.putText(img_display, f"Direction: {'FORWARD' if direction_forward else 'REVERSE'}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)  # Exibe direção do trajeto
            
            if paused:  # Se a navegação está pausada
                cv2.putText(img_display, "NAVIGATION PAUSED (press 'R' to continue)", 
                           (10, HEIGHT - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)  # Exibe mensagem de pausa
            
            cv2.putText(img_display, f"Speed factor: {speed_factor:.2f}", 
                       (WIDTH - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)  # Exibe fator de velocidade
            
            cv2.putText(img_display, "ENTER: Capture | T: New session | +/-: Speed | ESC: Exit", 
                       (10, HEIGHT - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)  # Exibe comandos disponíveis
            
            cv2.imshow('CARLA View', img_display)  # Exibe a imagem com informações na janela
            
            key = cv2.waitKey(1) & 0xFF  # Captura a tecla pressionada (atraso de 1ms)
            
            if key == 27:  # Se a tecla ESC for pressionada
                running = False  # Interrompe o loop
                print("Navigation interrupted by user.")  # Imprime mensagem de interrupção
                return False  # Retorna False para indicar interrupção
            elif key == ord('p') or key == ord('r'):  # Se tecla P (pausar) ou R (retomar) for pressionada
                paused = not paused if key == ord('p') else False  # Alterna pausa (P) ou retoma (R)
                print(f"Navigation {'paused' if paused else 'resumed'}.")  # Imprime estado da navegação
                if paused:  # Se pausado
                    vehicle_control.throttle = 0.0  # Remove aceleração
                    vehicle_control.brake = 1.0  # Aplica freio
                    vehicle.apply_control(vehicle_control)  # Aplica controle para parar o veículo
            elif key == ord('+') or key == ord('='):  # Se tecla + ou = for pressionada
                speed_factor = min(1.0, speed_factor + 0.05)  # Aumenta o fator de velocidade (máximo 1.0)
                print(f"Speed increased to: {speed_factor:.2f}")  # Imprime novo valor
            elif key == ord('-') or key == ord('_'):  # Se tecla - ou _ for pressionada
                speed_factor = max(0.1, speed_factor - 0.05)  # Diminui o fator de velocidade (mínimo 0.1)
                print(f"Speed decreased to: {speed_factor:.2f}")  # Imprime novo valor
    
    vehicle_control.throttle = 0.0  # Remove aceleração ao final
    vehicle_control.brake = 1.0  # Aplica freio
    vehicle.apply_control(vehicle_control)  # Para o veículo
    time.sleep(1)  # Aguarda 1 segundo para garantir que o veículo pare
    print("Vehicle stopped.")  # Imprime mensagem de parada
    return True  # Retorna True para indicar conclusão normal

def process_image(image):
    """Processa imagens RGB capturadas pela câmera do veículo."""
    global current_image  # Acessa a variável global para armazenar a imagem
    array = np.frombuffer(image.raw_data, dtype=np.uint8)  # Converte dados brutos da imagem em array NumPy
    current_image = np.reshape(array, (image.height, image.width, 4))[:, :, :3].copy()  # Reformata para (altura, largura, 4), remove canal alfa e copia

def process_segmentation_a(image):
    """Processa imagens de segmentação para criar uma máscara binária destacando faixas de estrada (CityScapes)."""
    global current_mask  # Acessa a variável global para armazenar a máscara
    image.convert(carla.ColorConverter.CityScapesPalette)  # Converte a imagem para a paleta CityScapes
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))[:, :, :3]  # Converte dados brutos para array RGB
    mask = np.all(arr == [157, 234, 50], axis=-1).astype(np.uint8) * 255  # Cria máscara binária (255 para faixas, 0 para outros)
    current_mask = mask  # Armazena a máscara

def process_segmentation(image):
    """Processa imagens de segmentação para criar uma máscara binária para a classe ID 24 (faixas de estrada)."""
    global current_mask  # Acessa a variável global para armazenar a máscara
    image.convert(carla.ColorConverter.Raw)  # Converte a imagem para formato bruto
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))  # Converte dados brutos para array
    class_ids = arr[:, :, 2]  # Extrai o canal azul (contém IDs de classe)
    unique_classes = np.unique(class_ids)  # Obtém IDs de classe únicos na imagem
    print("IDs de classes na imagem:", unique_classes)  # Imprime IDs para depuração
    mask = (class_ids == 24).astype(np.uint8) * 255  # Cria máscara binária (255 para ID 24, 0 para outros)
    current_mask = mask  # Armazena a máscara

def custom_colored_segmentation(image):
    """Processa imagens de segmentação para criar uma máscara colorida, destacando faixas de estrada (ID 24)."""
    global current_mask  # Acessa a variável global para armazenar a máscara
    image.convert(carla.ColorConverter.Raw)  # Converte a imagem para formato bruto
    raw = np.frombuffer(image.raw_data, dtype=np.uint8)  # Converte dados brutos em array NumPy
    seg = np.reshape(raw, (image.height, image.width, 4))[:, :, 2]  # Extrai o canal azul (IDs de classe)
    mask = np.zeros((image.height, image.width, 3), dtype=np.uint8)  # Cria uma máscara RGB vazia
    color_map = {  # Define mapeamento de IDs de classe para cores
        1:  (0, 0, 0),  # Estrada
        2:  (0, 0, 0),  # Calçada
        3:  (0, 0, 0),  # Prédio
        6:  (0, 0, 0),  # Linha de estrada
        8:  (0, 0, 0),  # Poste
        9:  (0, 0, 0),  # Semáforo
        11: (0, 0, 0),  # Vegetação
        14: (0, 0, 0),  # Carro
        18: (0, 0, 0),  # Placa de trânsito
        20: (0, 0, 0),  # Muro
        21: (0, 0, 0),  # Cerca
        22: (0, 0, 0),  # Guarda-corpo
        24: (255, 255, 255),  # Faixas de estrada (branco)
    }
    for class_id, color in color_map.items():  # Para cada ID de classe no mapeamento
        mask[seg == class_id] = color  # Aplica a cor correspondente aos pixels com esse ID
    current_mask = mask  # Armazena a máscara colorida

def main():
    """Configura a simulação CARLA, inicializa veículo/câmeras e gerencia navegação/captura de dados."""
    client = carla.Client('localhost', 2000)  # Conecta ao servidor CARLA na porta 2000
    client.set_timeout(10.0)  # Define timeout de 10 segundos para conexão
    world = client.load_world("/Game/Carla/Maps/Town10HD_Opt")  # Carrega o mapa Town10HD_Opt
    world = client.get_world()  # Obtém o objeto do mundo atual
    
    world.unload_map_layer(carla.MapLayer.ParkedVehicles)  # Desativa camada de veículos estacionados para otimização
    settings = world.get_settings()  # Obtém configurações do mundo
    settings.synchronous_mode = True  # Ativa modo síncrono
    settings.fixed_delta_seconds = 0.05  # Define intervalo fixo de 0.05s por tick
    world.apply_settings(settings)  # Aplica configurações ao mundo
    
    blueprint = world.get_blueprint_library()  # Obtém biblioteca de blueprints
    spawn_points = world.get_map().get_spawn_points()  # Obtém pontos de spawn do mapa
    vehicle_bp = blueprint.find('vehicle.tesla.model3')  # Seleciona blueprint do Tesla Model 3
    spawn_point = random.choice(spawn_points)  # Escolhe um ponto de spawn aleatório
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)  # Cria o veículo no ponto de spawn
    
    camera_transform = carla.Transform(carla.Location(x=2.0, z=1.4), carla.Rotation(pitch=-15))  # Define posição/rotação da câmera
    
    # Configura câmera RGB
    cam_rgb = blueprint.find('sensor.camera.rgb')  # Seleciona blueprint da câmera RGB
    cam_rgb.set_attribute('image_size_x', str(WIDTH))  # Define largura da imagem
    cam_rgb.set_attribute('image_size_y', str(HEIGHT))  # Define altura da imagem
    cam_rgb.set_attribute('fov', '90')  # Define campo de visão de 90 graus
    camera = world.spawn_actor(cam_rgb, camera_transform, attach_to=vehicle)  # Cria câmera RGB anexada ao veículo
    
    # Configura câmera de segmentação
    cam_seg = blueprint.find('sensor.camera.semantic_segmentation')  # Seleciona blueprint da câmera de segmentação
    cam_seg.set_attribute('image_size_x', str(WIDTH))  # Define largura da imagem
    cam_seg.set_attribute('image_size_y', str(HEIGHT))  # Define altura da imagem
    cam_seg.set_attribute('fov', '90')  # Define campo de visão de 90 graus
    seg = world.spawn_actor(cam_seg, camera_transform, attach_to=vehicle)  # Cria câmera de segmentação anexada ao veículo
    
    camera.listen(process_image)  # Associa a função process_image ao fluxo de dados da câmera RGB
    seg.listen(custom_colored_segmentation)  # Associa a função custom_colored_segmentation ao fluxo da câmera de segmentação
    
    vehicle.set_autopilot(True)  # Ativa o piloto automático do veículo inicialmente
    
    cv2.namedWindow("CARLA View", cv2.WINDOW_NORMAL)  # Cria uma janela OpenCV para exibir imagens
    for i in range(3, 0, -1):  # Contagem regressiva de 3 segundos
        print(f"Starting in {i}...")  # Imprime contagem
        time.sleep(1)  # Aguarda 1 segundo
        world.tick()  # Atualiza o mundo
    
    try:
        print("[INFO] Pressiona ENTER para capturar, ESC para sair")  # Exibe instruções iniciais
        direction_forward = True  # Define direção inicial como para frente
        current_speed = 0.5  # Define fator de velocidade inicial
        loop_count = 0  # Contador de loops completos (ida e volta)
        
        while True:  # Loop principal da simulação
            print("Finding road waypoints...")  # Imprime mensagem de busca de waypoints
            waypoints = get_road_waypoints(world, vehicle)  # Coleta waypoints a partir da posição do veículo
            
            if len(waypoints) < 10:  # Se menos de 10 waypoints forem encontrados
                print("Could not find enough waypoints. Repositioning vehicle...")  # Imprime aviso
                spawn_point = random.choice(spawn_points)  # Escolhe novo ponto de spawn
                time.sleep(1)  # Aguarda 1 segundo
                continue  # Tenta novamente
            
            if not drive_waypoints(world, vehicle, waypoints, direction_forward, current_speed):  # Executa navegação; retorna False se usuário interromper
                break  # Sai do loop se interrompido
            
            direction_forward = not direction_forward  # Alterna direção para próxima iteração
            loop_count += 0.5  # Incrementa contador (0.5 por direção)
            print(f"\nRoute completed! Automatically switching direction to {('FORWARD' if direction_forward else 'REVERSE')}")  # Imprime mensagem de troca de direção
    
    finally:
        camera.stop()  # Para o fluxo de dados da câmera RGB
        seg.stop()  # Para o fluxo de dados da câmera de segmentação
        camera.destroy()  # Destrói a câmera RGB
        seg.destroy()  # Destrói a câmera de segmentação
        vehicle.destroy()  # Destrói o veículo
        world.apply_settings(carla.WorldSettings(synchronous_mode=False))  # Desativa modo síncrono
        cv2.destroyAllWindows()  # Fecha todas as janelas OpenCV
        print(f"[INFO] Sessão terminada. {frame_count} imagens guardadas.")  # Imprime número de imagens salvas

if __name__ == "__main__":
    main()  # Executa a função principal se o script for rodado diretamente