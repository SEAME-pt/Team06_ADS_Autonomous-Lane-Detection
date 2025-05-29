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
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=20, minLineLength=20, maxLineGap=15)
    
    if lines is not None and len(lines) > 0:
        x_sum, y_sum, angles, count = 0, 0, [], 0
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
            print(f"Detecção: x_center={x_center:.2f}, y_ref={y_ref:.3f} m, psi_ref={psi_ref:.3f} rad, count={count}")
            # Visualizar linhas detectadas
            line_display = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
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