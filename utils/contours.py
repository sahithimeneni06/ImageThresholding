import numpy as np
import cv2
from PIL import Image

def detect_contours(img, thresh_img):
    contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contoured = cv2.drawContours(np.array(img).copy(), contours, -1, (0, 255, 0), 2)
    return contoured, len(contours)
