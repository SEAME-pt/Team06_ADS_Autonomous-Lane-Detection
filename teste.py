import cv2




width = 1280  
height = 720 
framerate = 30  
flip_method = 0  #  angls


pipeline = (
    f"nvarguscamerasrc ! "
    f"video/x-raw(memory:NVMM), width=(int){width}, height=(int){height}, format=(string)NV12, framerate=(fraction){framerate}/1 ! "
    f"nvvidconv flip-method={flip_method} ! "
    f"video/x-raw, width=(int){width}, height=(int){height}, format=(string)BGRx ! "
    f"videoconvert ! "
    f"video/x-raw, format=(string)BGR ! appsink"
)

# Inicializa a captura de vídeo
camera = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

print(cv2.getBuildInformation())

if cv2.cuda.getCudaEnabledDeviceCount() > 0:
    print("CUDA está habilitado no OpenCV!")
else:
    print("CUDA não está habilitado no OpenCV.")

if not camera.isOpened():
    print("Erro ao abrir a câmera com OpenCV.")
    exit()

print(" 'q' para sair.")

while True:
    ret, frame = camera.read()

    if not ret:
        print("Falha ao capturar o frame.")
        break

    cv2.imshow("Câmera Jetson Nano", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
