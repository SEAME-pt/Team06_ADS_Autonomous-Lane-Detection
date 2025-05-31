import cv2
import numpy as np
import math

# [Outras funções (trace_line, order_line_pixels, align_lines, merge_lines, resample_line, calculate_center_line) permanecem as mesmas]
def trace_line(roi, start_x, start_y, visited, max_gap=2):
    """Rastrea uma linha a partir de um pixel branco, seguindo pixels adjacentes com prioridade diagonal, tolerando pequenos gaps."""
    line_pixels = [(start_x, start_y)]
    visited[start_y, start_x] = True
    stack = [(start_x, start_y)]
    
    while stack:
        x, y = stack.pop()
        # Procurar em uma janela 5x5 para tolerar pequenos gaps
        for dy in range(-max_gap, max_gap + 1):
            for dx in range(-max_gap, max_gap + 1):
                if dx == 0 and dy == 0:
                    continue
                new_x, new_y = x + dx, y + dy
                if (0 <= new_x < roi.shape[1] and 0 <= new_y < roi.shape[0] and
                    not visited[new_y, new_x] and roi[new_y, new_x] == 255):
                    line_pixels.append((new_x, new_y))
                    visited[new_y, new_x] = True
                    stack.append((new_x, new_y))
    return line_pixels

def order_line_pixels(line_pixels):
    """Ordena os pontos de uma linha com base no valor de y (de menor para maior)."""
    if len(line_pixels) < 2:
        return line_pixels
    # Ordenar por y, e para y iguais, por x
    sorted_pixels = sorted(line_pixels, key=lambda p: (p[1], p[0]))
    return sorted_pixels

def align_lines(line1_pixels, line2_pixels):
    """Alinha as direções das duas linhas para que sigam a mesma orientação (de menor y para maior y)."""
    # Verificar os pontos inicial e final de cada linha
    start1_y = line1_pixels[0][1]
    end1_y = line1_pixels[-1][1]
    start2_y = line2_pixels[0][1]
    end2_y = line2_pixels[-1][1]

    # Se a linha 2 estiver na direção oposta (começando com y maior), invertê-la
    if start2_y > end2_y:
        line2_pixels = list(reversed(line2_pixels))

    # Garantir que ambas as linhas comecem com o menor y
    if start1_y > end1_y:
        line1_pixels = list(reversed(line1_pixels))

    return line1_pixels, line2_pixels

def merge_lines(lines, max_distance=5):
    """Junta linhas que estão próximas umas das outras (extremidades a poucos pixels de distância)."""
    if not lines:
        return lines

    merged_lines = [lines[0]]
    for i in range(1, len(lines)):
        current_line = lines[i]
        merged = False
        for j in range(len(merged_lines)):
            prev_line = merged_lines[j]
            # Verificar distância entre extremos
            start1, end1 = prev_line[0], prev_line[-1]
            start2, end2 = current_line[0], current_line[-1]
            distances = [
                math.sqrt((start1[0] - start2[0])**2 + (start1[1] - start2[1])**2),
                math.sqrt((start1[0] - end2[0])**2 + (start1[1] - end2[1])**2),
                math.sqrt((end1[0] - start2[0])**2 + (end1[1] - start2[1])**2),
                math.sqrt((end1[0] - end2[0])**2 + (end1[1] - end2[1])**2)
            ]
            min_dist = min(distances)
            if min_dist <= max_distance:
                # Juntar as linhas
                if distances.index(min_dist) == 0:  # start1 -> start2
                    merged_lines[j] = list(reversed(current_line)) + prev_line
                elif distances.index(min_dist) == 1:  # start1 -> end2
                    merged_lines[j] = current_line + prev_line
                elif distances.index(min_dist) == 2:  # end1 -> start2
                    merged_lines[j] = prev_line + current_line
                else:  # end1 -> end2
                    merged_lines[j] = prev_line + list(reversed(current_line))
                merged = True
                break
        if not merged:
            merged_lines.append(current_line)
    return merged_lines

def resample_line(line_pixels, num_points=50):
    """Reamostra uma linha para ter um número fixo de pontos, interpolando linearmente com base na distância acumulada."""
    if len(line_pixels) < 2:
        return line_pixels

    # Calcular distâncias acumuladas
    distances = [0]
    total_length = 0
    for i in range(1, len(line_pixels)):
        x1, y1 = line_pixels[i-1]
        x2, y2 = line_pixels[i]
        dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        total_length += dist
        distances.append(total_length)

    if total_length == 0:
        return line_pixels

    # Normalizar as distâncias
    distances = [d / total_length for d in distances]

    # Gerar novos pontos
    new_points = []
    step = 1.0 / (num_points - 1)
    for i in range(num_points):
        t = i * step
        # Encontrar o intervalo onde t está
        for j in range(len(distances) - 1):
            if distances[j] <= t <= distances[j + 1]:
                frac = (t - distances[j]) / (distances[j + 1] - distances[j])
                x1, y1 = line_pixels[j]
                x2, y2 = line_pixels[j + 1]
                x = x1 + frac * (x2 - x1)
                y = y1 + frac * (y2 - y1)
                new_points.append((int(x), int(y)))
                break
        if t >= distances[-1]:  # Último ponto
            new_points.append(line_pixels[-1])

    return new_points

def calculate_center_line(line1_pixels, line2_pixels, num_points=50):
    """Calcula a linha central reamostrando e alinhando as duas linhas com base na distância acumulada, com suavização."""
    if not line1_pixels or not line2_pixels:
        return []

    # Ordenar os pontos de cada linha por y
    line1_pixels = order_line_pixels(line1_pixels)
    line2_pixels = order_line_pixels(line2_pixels)

    # Alinhar as direções das linhas
    line1_pixels, line2_pixels = align_lines(line1_pixels, line2_pixels)

    # Reamostrar ambas as linhas
    line1_resampled = resample_line(line1_pixels, num_points)
    line2_resampled = resample_line(line2_pixels, num_points)

    # Calcular a linha central como a média dos pontos correspondentes
    center_line = []
    min_length = min(len(line1_resampled), len(line2_resampled))
    for i in range(min_length):
        x1, y1 = line1_resampled[i]
        x2, y2 = line2_resampled[i]
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        center_line.append((int(center_x), int(center_y)))

    # Suavizar a linha central com média móvel (janela de 5 pontos)
    smoothed_center_line = []
    window_size = 15
    for i in range(len(center_line)):
        if i < window_size - 1:
            # Para os primeiros pontos, usar média parcial
            window = center_line[:i + 1]
        elif i >= len(center_line) - (window_size - 1):
            # Para os últimos pontos, usar média parcial
            window = center_line[i - (window_size - 1):]
        else:
            # Janela completa
            window = center_line[i - (window_size - 1):i + 1]
        avg_x = int(sum(p[0] for p in window) / len(window))
        avg_y = int(sum(p[1] for p in window) / len(window))
        smoothed_center_line.append((avg_x, avg_y))

    return smoothed_center_line

# [Outras funções (trace_line, order_line_pixels, align_lines, merge_lines, resample_line, calculate_center_line) permanecem as mesmas]

def process_mask(mask_path, nmpc_pred=None, current_state=None):
    """Processa uma imagem de máscara para identificar linhas brancas apenas na ROI e exibe previsões NMPC."""
    # Carregar a imagem
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print("Erro: Não foi possível carregar a imagem.")
        return None, None, None, False, []

    # Definir ROI
    height, width = mask.shape
    roi_x = int(width * 0.05)
    roi_width = int(width * 0.90)
    roi_y = int(height * 0.5)
    roi_height = int(height * 0.6)
    roi = mask[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]
    print(f"ROI dimensions: width={roi_width}, height={roi_height}")

    # Encontrar pixels brancos (valor 255) dentro da ROI
    white_pixels = np.column_stack(np.where(roi == 255))
    if len(white_pixels) == 0:
        print("Nenhum pixel branco detectado na ROI.")
        cv2.imshow('Máscara sem Linhas', roi)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return None, None, roi, False, []

    # Rastrear todas as linhas a partir de pixels brancos não visitados dentro da ROI
    visited = np.zeros_like(roi, dtype=bool)
    lines = []
    for y, x in white_pixels:
        if not visited[y, x]:
            line_pixels = trace_line(roi, x, y, visited, max_gap=2)
            if len(line_pixels) > 10:  # Filtrar linhas muito curtas
                lines.append(line_pixels)

    # Juntar linhas que estão conectadas ou muito próximas
    lines = merge_lines(lines, max_distance=5)

    # Verificar se pelo menos uma linha foi detectada
    if not lines:
        print("Nenhuma linha detectada a partir dos pixels brancos na ROI.")
        cv2.imshow('Máscara sem Linhas', roi)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return None, None, roi, False, []

    # Calcular a linha central ao longo das duas linhas
    center_line = []
    if len(lines) >= 2:
        line1_pixels = lines[0]  # Linha esquerda
        line2_pixels = lines[1]  # Linha direita
        center_line = calculate_center_line(line1_pixels, line2_pixels, num_points=50)

    # Imprimir os pontos da linha central para depuração
    print("Center line points (x, y):")
    for x, y in center_line[:5]:  # Mostrar os primeiros 5 pontos para não sobrecarregar
        print(f"  ({x}, {y})")

    # Visualizar linhas detectadas com cores diferentes apenas na ROI
    line_display = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
    colors = [(0, 255, 0), (0, 0, 255)]
    for idx, line_pixels in enumerate(lines[:2]):  # Usar apenas as duas primeiras linhas
        color = colors[idx % len(colors)]
        for i in range(len(line_pixels) - 1):
            x1, y1 = line_pixels[i]
            x2, y2 = line_pixels[i + 1]
            cv2.line(line_display, (x1, y1), (x2, y2), color, 2)
    if center_line:
        for i in range(len(center_line) - 1):
            x1, y1 = center_line[i]
            x2, y2 = center_line[i + 1]
            cv2.line(line_display, (x1, y1), (x2, y2), (255, 255, 255), 2)  # Linha central branca

    # Desenhar a trajetória predita pelo NMPC
    if nmpc_pred is not None and current_state is not None:
        # Extrair as posições preditas (x, y) do X_pred
        x_pred = nmpc_pred[0, :].flatten()  # Converter para array 1D
        y_pred = nmpc_pred[1, :].flatten()  # Converter para array 1D
        x_current, y_current = current_state[0], current_state[1]  # Posição atual

        # Imprimir valores para depuração
        print("Predicted trajectory points (x_pred, y_pred):")
        for i in range(len(x_pred)):
            print(f"  Step {i}: ({x_pred[i]}, {y_pred[i]})")
        print(f"Current position: x={x_current}, y={y_current}")

        # Desenhar a trajetória predita como uma linha tracejada (vermelha)
        for i in range(len(x_pred) - 1):
            x1 = int(x_pred[i])
            y1 = int(y_pred[i])
            x2 = int(x_pred[i + 1])
            y2 = int(y_pred[i + 1])
            # Verificar se as coordenadas estão dentro da ROI
            if (0 <= x1 < roi_width and 0 <= y1 < roi_height) and (0 <= x2 < roi_width and 0 <= y2 < roi_height):
                cv2.line(line_display, (x1, y1), (x2, y2), (0, 0, 255), 2, cv2.LINE_4)  # Linha tracejada
            else:
                print(f"Warning: Predicted point out of ROI bounds at step {i}: ({x1}, {y1}) to ({x2}, {y2})")

        # Desenhar um marcador na posição atual (verde)
        if 0 <= int(x_current) < roi_width and 0 <= int(y_current) < roi_height:
            cv2.circle(line_display, (int(x_current), int(y_current)), 5, (0, 255, 0), -1)
        else:
            print(f"Warning: Current position out of ROI bounds: ({x_current}, {y_current})")

    cv2.imshow('Linhas Detectadas', line_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 0.0, 0.0, line_display, True if center_line else False, center_line

def main():
    """Função principal para testar a detecção de linhas."""
    mask_path = "../mask/mask_test03.png"
    y_ref, psi_ref, display_img, lines_detected, center_line = process_mask(mask_path)
    if not lines_detected:
        print("Nenhuma linha detectada com sucesso.")
    return center_line

if __name__ == "__main__":
    center_line = main()