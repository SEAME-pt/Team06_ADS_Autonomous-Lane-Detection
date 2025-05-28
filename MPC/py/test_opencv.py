import cv2
import numpy as np
img = np.zeros((300, 400, 3), dtype=np.uint8)
img[100:200, 100:300] = (0, 255, 0)  # Retângulo verde
cv2.imshow("Test", img)
cv2.waitKey(0)
cv2.destroyAllWindows()