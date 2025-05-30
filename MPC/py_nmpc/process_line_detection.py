import cv2
import numpy as np
import math

def trace_line(roi, start_x, start_y, visited):
    """Rastrea uma linha a partir de um pixel branco, seguindo pixels adjacentes com prioridade diagonal."""
    line_pixels = [(start_x, start_y)]
    visited[start_y, start_x] = True
    stack = [(start_x, start_y)]
    
    while stack:
        x, y = stack.pop()
        # Priorizar direções diagonais
        directions = [(-1, -1), (1, 1), (-1, 1), (1, -1),  # Diagonais
                      (0, -1), (0, 1), (-1, 0), (1, 0)]  # Horizontais/verticais
        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy
            if (0 <= new_x < roi.shape[1] and 0 <= new_y < roi.shape[0] and
                not visited[new_y, new_x] and roi[new_y, new_x] == 255):
                line_pixels.append((new_x, new_y))
                visited[new_y, new_x] = True
                stack.append((new_x, new_y))
    return line_pixels

def process_mask(mask_path):
    """Processa uma imagem de máscara para identificar linhas brancas apenas na ROI e exibir coordenadas."""
    # Carregar a imagem
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print("Erro: Não foi possível carregar a imagem.")
        return None, None, None, False

    # Definir ROI
    height, width = mask.shape
    roi_x = int(width * 0.15)
    roi_width = int(width * 0.7)
    roi_y = int(height * 0.4)
    roi_height = int(height * 0.7)
    roi = mask[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]

    # Encontrar pixels brancos (valor 255) dentro da ROI
    white_pixels = np.column_stack(np.where(roi == 255))
    if len(white_pixels) == 0:
        print("Nenhum pixel branco detectado na ROI.")
        cv2.imshow('Máscara sem Linhas', roi)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return 0.0, 0.0, roi, False

    # Rastrear todas as linhas a partir de pixels brancos não visitados dentro da ROI
    visited = np.zeros_like(roi, dtype=bool)
    lines = []
    for y, x in white_pixels:
        if not visited[y, x]:
            line_pixels = trace_line(roi, x, y, visited)
            if len(line_pixels) > 10:  # Filtrar linhas muito curtas
                lines.append(line_pixels)
                print(f"Pixels rastreados para nova linha: {line_pixels}")  # Depuração

    # Verificar se pelo menos uma linha foi detectada
    if not lines:
        print("Nenhuma linha detectada a partir dos pixels brancos na ROI.")
        cv2.imshow('Máscara sem Linhas', roi)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return 0.0, 0.0, roi, False

    # Calcular pontos extremos e médio para cada linha dentro da ROI
    processed_lines = []
    mid_points = []  # Armazena o ponto médio de cada linha
    print("\nCoordenadas das linhas (relativas à ROI, de 0 a width/height da ROI):")
    # Ordenar linhas por x médio (esquerda para direita)
    lines_with_avg_x = [(sum(p[0] for p in line) / len(line), line) for line in lines]
    lines_with_avg_x.sort()  # Ordenar por x médio
    lines = [line for _, line in lines_with_avg_x]

    for idx, line_pixels in enumerate(lines):
        # Determinar se é linha esquerda ou direita
        is_left_line = idx == 0  # Primeira linha é a mais à esquerda

        if is_left_line:
            # Linha esquerda: mais próximo (menor y, maior x), mais afastado (maior y, maior x)
            y_sorted_min = sorted(line_pixels, key=lambda p: (p[1], -p[0]))  # Menor y, maior x
            y_sorted_max = sorted(line_pixels, key=lambda p: (-p[1], -p[0]))  # Maior y, maior x
            near_point = y_sorted_min[0]
            far_point = y_sorted_max[0]
        else:
            # Linha direita: mais próximo (menor y, menor x), mais afastado (maior y, menor x)
            y_sorted_min = sorted(line_pixels, key=lambda p: (p[1], p[0]))  # Menor y, menor x
            y_sorted_max = sorted(line_pixels, key=lambda p: (-p[1], p[0]))  # Maior y, menor x
            near_point = y_sorted_min[0]
            far_point = y_sorted_max[0]

        # Usar near_point e far_point para desenho
        x1, y1 = near_point
        x2, y2 = far_point
        # Ponto médio da linha
        mid_x = (near_point[0] + far_point[0]) / 2
        mid_y = (near_point[1] + far_point[1]) / 2
        mid_points.append((mid_x, mid_y))
        # Exibir coordenadas
        near_x, near_y = near_point
        far_x, far_y = far_point
        print(f"Linha {idx + 1}:")
        print(f"  Ponto mais afastado: (x={far_x:.2f}, y={far_y:.2f})")
        print(f"  Ponto médio: (x={mid_x:.2f}, y={mid_y:.2f})")
        print(f"  Ponto mais próximo: (x={near_x:.2f}, y={near_y:.2f})")
        processed_lines.append(((x1, y1), (x2, y2)))

    # Calcular a linha central ao longo das duas linhas
    center_line = []
    if len(lines) >= 2:
        line1_pixels = lines[0]  # Linha esquerda
        line2_pixels = lines[1]  # Linha direita

        # Extrair todos os y únicos e seus x correspondentes
        y_values = sorted(set(y for _, y in line1_pixels).intersection(y for _, y in line2_pixels))
        for y in y_values:
            x1s = [x for x, y1 in line1_pixels if y1 == y]
            x2s = [x for x, y2 in line2_pixels if y2 == y]
            if x1s and x2s:  # Apenas incluir y que existe em ambas as linhas
                center_x = (sum(x1s) / len(x1s) + sum(x2s) / len(x2s)) / 2
                center_line.append((int(center_x), y))

        # Conectar os pontos da linha central
        #print("\nLinha central (pontos médios ao longo de y):")
        #for x, y in center_line:
        #    print(f"  (x={x:.2f}, y={y:.2f})")

    # Visualizar linhas detectadas com cores diferentes apenas na ROI
    line_display = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
    colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0), (0, 255, 255),
              (255, 0, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0)]
    for idx, ((x1, y1), (x2, y2)) in enumerate(processed_lines):
        color = colors[idx % len(colors)]  # Ciclo de cores
        cv2.line(line_display, (x1, y1), (x2, y2), color, 2)
    # Desenhar a linha central
    if center_line:
        for i in range(len(center_line) - 1):
            x1, y1 = center_line[i]
            x2, y2 = center_line[i + 1]
            cv2.line(line_display, (x1, y1), (x2, y2), (255, 255, 255), 2)  # Linha branca
    cv2.imshow('Linhas Detectadas', line_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 0.0, 0.0, line_display, True

def main():
    """Função principal para testar a detecção de linhas."""
    mask_path = "../mask/mask_test03.png"
    y_ref, psi_ref, display_img, lines_detected = process_mask(mask_path)
    if not lines_detected:
        print("Nenhuma linha detectada com sucesso.")

if __name__ == "__main__":
    main()