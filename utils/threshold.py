import numpy as np
import cv2
from PIL import Image

def glob_thresh(img, thresh_value, th_type):
    img_np = np.array(img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    _, thresh_img = cv2.threshold(gray, thresh_value, 255, th_type)
    return thresh_img

def adap_thresh(img, adaMethod, threshType, blockSize, C=2):
    img_np = np.array(img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    thresh_img = cv2.adaptiveThreshold(gray, 255, adaMethod, threshType, blockSize, C)
    return thresh_img