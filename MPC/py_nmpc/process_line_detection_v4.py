import cv2
import numpy as np
import math

def process_mask(mask_path):
    """Processa uma imagem de máscara para extrair y_ref e psi_ref com debug."""
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

    edges = cv2.Canny(roi, 100, 200)  # Aumentar thresholds para reduzir ruído
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=200, minLineLength=150, maxLineGap=20)
    
    if lines is not None and len(lines) > 0:
        print(f"Detectadas {len(lines)} linhas totais.")
        x_sum, y_sum, angles, count = 0, 0, [], 0
        # Calcular o centro antes do filtro de ângulo
        x_sum_total, y_sum_total, count_total = 0, 0, 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            x_sum_total += (x1 + x2) / 2
            y_sum_total += (y1 + y2) / 2
            count_total += 1
        if count_total > 0:
            x_center_total = x_sum_total / count_total + roi_x
            y_center_total = y_sum_total / count_total + roi_y
            print(f"Centro das linhas totais: x_center={x_center_total:.2f}, y_center={y_center_total:.2f}")

        # Exibir coordenadas de cada linha (mais afastado, médio, mais próximo)
        print("\nCoordenadas das linhas (relativas à imagem completa):")
        for idx, line in enumerate(lines):
            x1, y1, x2, y2 = line[0]
            # Calcular distâncias à origem (0, 0) da ROI
            dist1 = math.sqrt(x1**2 + y1**2)
            dist2 = math.sqrt(x2**2 + y2**2)
            # Determinar pontos mais afastado e mais próximo
            if dist1 > dist2:
                far_point = (x1, y1)
                near_point = (x2, y2)
            else:
                far_point = (x2, y2)
                near_point = (x1, y1)
            # Calcular ponto médio
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

        # Aplicar o filtro de ângulo
        for line in lines:
            x1, y1, x2, y2 = line[0]
            x_sum += (x1 + x2) / 2
            y_sum += (y1 + y2) / 2
            angle = math.atan2(y2 - y1, x2 - x1)
            if abs(angle - np.pi / 2) > np.deg2rad(10):  # Filtrar linhas muito verticais
                angles.append(angle)
                count += 1
        if count > 0:
            x_center = x_sum / count + roi_x
            y_center = y_sum / count + roi_y
            y_ref = (x_center - width / 2) * 0.005
            y_ref_history = [y_ref]  # Simulação de histórico
            y_ref = y_ref  # Sem média móvel por agora
            psi_ref = np.mean(angles) - np.pi / 2 if angles else 0.0
            psi_ref = np.clip(psi_ref, -np.pi / 4, np.pi / 4)
            print(f"\nDetectadas {count} linhas após filtro de ângulo.")
            print(f"Centro das linhas filtradas: x_center={x_center:.2f}, y_center={y_center:.2f}")
            print(f"Detecção: x_center={x_center:.2f}, y_ref={y_ref:.3f} m, psi_ref={psi_ref:.3f} rad, count={count}")
            # Visualizar linhas detectadas com cores diferentes
            line_display = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
            colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0), (0, 255, 255),
                      (255, 0, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
                      (0, 128, 128), (128, 0, 128), (255, 165, 0), (173, 216, 230), (255, 192, 203)]
            for idx, line in enumerate(lines):
                x1, y1, x2, y2 = line[0]
                color = colors[idx % len(colors)]  # Ciclo de cores se mais linhas que cores
                cv2.line(line_display, (x1, y1), (x2, y2), color, 2)
            cv2.imshow('Linhas Detectadas', line_display)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return y_ref, psi_ref, line_display, True
    print("Nenhuma linha válida detectada")
    cv2.imshow('Máscara sem Linhas', roi)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 0.0, 0.0, roi, False

def main():
    """Função principal para testar a detecção de linhas."""
    mask_path = "../mask/mask_test01.png"  # Substitua pelo caminho real da imagem
    y_ref, psi_ref, display_img, lines_detected = process_mask(mask_path)
    if lines_detected:
        print(f"Resultados: y_ref = {y_ref:.3f} m, psi_ref = {psi_ref:.3f} rad")
    else:
        print("Nenhuma linha detectada com sucesso.")

if __name__ == "__main__":
    main()