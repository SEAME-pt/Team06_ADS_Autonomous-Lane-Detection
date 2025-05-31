import mask_line_detection
import nmpc_controller

def main():
    # Obter a linha central do script de detecção
    center_line = mask_line_detection.main()
    if not center_line:
        print("Falha ao obter a linha central.")
        return

    # Executar NMPC e obter previsões e estado atual
    X_pred, state_init = nmpc_controller.main(center_line)
    if X_pred is None:
        print("Falha ao executar o NMPC.")
        return

    # Reprocessar a imagem com as previsões NMPC
    mask_path = "../mask/mask_test01.png"
    y_ref, psi_ref, display_img, lines_detected, _ = mask_line_detection.process_mask(mask_path, X_pred, state_init)

if __name__ == "__main__":
    main()