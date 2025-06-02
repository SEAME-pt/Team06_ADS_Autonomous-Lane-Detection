import cv2
import numpy as np
import os
from pathlib import Path

def process_image(image_path, output_path, gray_color=(128, 128, 128), border_color=(64, 64, 64), border_thickness=3):
    """
    Processa uma imagem convertendo áreas brancas para cinza e adicionando contorno.
    
    Args:
        image_path: caminho da imagem original
        output_path: caminho para salvar a imagem processada
        gray_color: cor cinza para substituir o branco (B, G, R)
        border_color: cor do contorno (B, G, R)
        border_thickness: espessura do contorno em pixels
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Erro: Não foi possível carregar a imagem {image_path}")
        return False
    
    # Converter para HSV para melhor detecção de cores
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Definir range para detectar áreas brancas/claras
    # Ajuste estes valores se necessário
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    
    # Criar máscara para áreas brancas
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # Aplicar a cor cinza nas áreas brancas
    img[white_mask > 0] = gray_color
    
    # Criar contorno das formas
    # Encontrar contornos na máscara
    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Desenhar contorno ao redor das formas
    cv2.drawContours(img, contours, -1, border_color, border_thickness)
    
    # Salvar a imagem processada
    cv2.imwrite(output_path, img)
    return True

def process_folder(input_folder, output_folder, gray_color=(128, 128, 128), border_color=(64, 64, 64), border_thickness=3):
    """
    Processa todas as imagens de uma pasta.
    
    Args:
        input_folder: pasta com as imagens originais
        output_folder: pasta onde salvar as imagens processadas
        gray_color: cor cinza para substituir o branco (B, G, R)
        border_color: cor do contorno (B, G, R)
        border_thickness: espessura do contorno em pixels
    """
    # Criar pasta de saída se não existir
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Extensões de imagem suportadas
    supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    processed_count = 0
    
    # Processar cada arquivo na pasta
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        
        # Verificar se é um arquivo de imagem
        if os.path.isfile(file_path) and Path(filename).suffix.lower() in supported_extensions:
            output_path = os.path.join(output_folder, filename)
            
            print(f"Processando: {filename}")
            
            if process_image(file_path, output_path, gray_color, border_color, border_thickness):
                processed_count += 1
                print(f"✓ Processado com sucesso: {filename}")
            else:
                print(f"✗ Erro ao processar: {filename}")
    
    print(f"\nProcessamento concluído!")
    print(f"Total de imagens processadas: {processed_count}")

def main():
    """Função principal - configure aqui os caminhos das pastas"""
    
    input_folder = "/home/djoker/code/conda/TwinLiteNet/TwinLiteNet/organized_dataset/val/segments"     # Pasta com as imagens originais
    output_folder = "imagens_processadas"  # Pasta onde salvar as imagens processadas
    
    gray_color = (128, 128, 128)    # Cor cinza 
    border_color = (220, 220, 220)     # Cor do contorno 
    border_thickness = 2            # Espessura do contorno em pixels
    
    if not os.path.exists(input_folder):
        print(f"Erro: A pasta '{input_folder}' não existe!")
        print("Crie a pasta e coloque as imagens nela, ou altere o caminho no script.")
        return
    
    print(f"Processando imagens da pasta: {input_folder}")
    print(f"Salvando imagens na pasta: {output_folder}")
    print(f"Cor cinza: {gray_color}")
    print(f"Cor do contorno: {border_color}")
    print(f"Espessura do contorno: {border_thickness}px")
    print("-" * 50)
    
 
    process_folder(input_folder, output_folder, gray_color, border_color, border_thickness)

if __name__ == "__main__":
    main()
 