import cv2
import numpy as np
import math

def trace_line(roi, start_x, start_y, visited):
    """Rastrea uma linha a partir de um pixel branco, seguindo pixels adjacentes."""
    line_pixels = [(start_x, start_y)]
    visited[start_y, start_x] = True
    stack = [(start_x, start_y)]
    
    while stack:
        x, y = stack.pop()
        # Verificar 8 vizinhos (connectivity 8)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                new_x, new_y = x + dx, y + dy
                if (0 <= new_x < roi.shape[1] and 0 <= new_y < roi.shape[0] and
                    not visited[new_y, new_x] and roi[new_y, new_x] == 255):
                    line_pixels.append((new_x, new_y))
                    visited[new_y, new_x] = True
                    stack.append((new_x, new_y))
    return line_pixels

def process_mask(mask_path):
    """Processa uma imagem de máscara para identificar linhas brancas e exibir coordenadas."""
    # Carregar a imagem
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print("Erro: Não foi possível carregar a imagem.")
        return None, None, None, False

    # Definir ROI
    height, width = mask.shape
    roi_x = int(width * 0.1)
    roi_width = int(width * 0.8)
    roi_y = int(height * 0.5)
    roi_height = int(height * 0.6)
    roi = mask[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]

    # Encontrar pixels brancos (valor 255)
    white_pixels = np.column_stack(np.where(roi == 255))
    if len(white_pixels) == 0:
        print("Nenhum pixel branco detectado na ROI.")
        cv2.imshow('Máscara sem Linhas', roi)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return 0.0, 0.0, roi, False

    # Rastrear linhas a partir de pixels brancos não visitados
    visited = np.zeros_like(roi, dtype=bool)
    lines = []
    for y, x in white_pixels:
        if not visited[y, x]:
            line_pixels = trace_line(roi, x, y, visited)
            if len(line_pixels) > 10:  # Filtrar linhas muito curtas
                lines.append(line_pixels)

    # Limitar a 2 linhas (as duas principais)
    lines = lines[:2]
    if len(lines) < 2:
        print("Menos de 2 linhas detectadas a partir dos pixels brancos.")
        cv2.imshow('Máscara sem Linhas', roi)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return 0.0, 0.0, roi, False

    # Calcular pontos extremos e médio para cada linha
    processed_lines = []
    print("\nCoordenadas das linhas (relativas à imagem completa):")
    for idx, line_pixels in enumerate(lines):
        x_coords = [p[0] for p in line_pixels]
        y_coords = [p[1] for p in line_pixels]
        x1, x2 = min(x_coords), max(x_coords)
        y1, y2 = min(y_coords), max(y_coords)
        # Ponto mais próximo (menor distância à origem da ROI)
        dist1 = math.sqrt(x1**2 + y1**2)
        dist2 = math.sqrt(x2**2 + y2**2)
        if dist1 < dist2:
            near_point = (x1, y1)
            far_point = (x2, y2)
        else:
            near_point = (x2, y2)
            far_point = (x1, y1)
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        # Ajustar coordenadas para a imagem completa
        far_x, far_y = far_point[0] + roi_x, far_point[1] + roi_y
        near_x, near_y = near_point[0] + roi_x, near_point[1] + roi_y
        mid_x_adjusted, mid_y_adjusted = mid_x + roi_x, mid_y + roi_y
        print(f"Linha {idx + 1}:")
        print(f"  Ponto mais afastado: (x={far_x:.2f}, y={far_y:.2f})")
        print(f"  Ponto médio: (x={mid_x_adjusted:.2f}, y={mid_y_adjusted:.2f})")
        print(f"  Ponto mais próximo: (x={near_x:.2f}, y={near_y:.2f})")
        processed_lines.append(((x1, y1), (x2, y2)))

    # Visualizar linhas detectadas com cores diferentes
    line_display = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
    colors = [(0, 255, 0), (0, 0, 255)]  # Verde e Azul para as 2 linhas
    for idx, ((x1, y1), (x2, y2)) in enumerate(processed_lines):
        color = colors[idx % len(colors)]
        cv2.line(line_display, (x1, y1), (x2, y2), color, 2)
    cv2.imshow('Linhas Detectadas', line_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 0.0, 0.0, line_display, True

def main():
    """Função principal para testar a detecção de linhas."""
    mask_path = "../mask/mask_test02.png"  # Substitua pelo caminho real da imagem
    y_ref, psi_ref, display_img, lines_detected = process_mask(mask_path)
    if not lines_detected:
        print("Nenhuma linha detectada com sucesso.")

if __name__ == "__main__":
    main()