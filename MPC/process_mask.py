import cv2
import numpy as np

def calcular_deslocamento_lateral(mascara):
    # Máscara: imagem binária 256x128, branco (255) para linhas, preto (0) para fundo
    # Pegar a linha inferior (y=128) ou próxima (y=120 para evitar bordas)
    linha = mascara[300, :]  # Linha y=120

    # Encontrar pixels brancos (linhas)
    pixels_brancos = np.where(linha == 255)[0]  # Índices dos pixels brancos

    if len(pixels_brancos) == 0:
        return 0  # Retornar 0 se não encontrar linhas

    # Linha esquerda: menor x; linha direita: maior x
    x_esquerda = pixels_brancos.min()
    x_direita = pixels_brancos.max()

    # Calcular centro
    x_centro = (x_esquerda + x_direita) / 2

    # Centro da imagem (alinhado com o veículo)
    x_centro_imagem = 640 / 2  # 128 pixels

    # Deslocamento lateral (em pixels)
    deslocamento = x_centro - x_centro_imagem

    print(f"Linha esquerda: x={x_esquerda}, Linha direita: x={x_direita}")
    print(f"Centro da trajetória: x={x_centro}")
    print(f"Deslocamento lateral: {deslocamento} pixels")

    return deslocamento

# Exemplo de uso
mascara = cv2.imread('mask/mask_test01.png', cv2.IMREAD_GRAYSCALE)  # Carregar máscara
deslocamento = calcular_deslocamento_lateral(mascara)
print(f"Deslocamento lateral: {deslocamento} pixels")