import numpy as np
import cv2

# VIDEO_SRC = '161905'
VIDEO_SRC = '161439'
cap = cv2.VideoCapture(VIDEO_SRC + '.mp4')

while(cap.isOpened()):
    end_flag, frame = cap.read()

    # 各ピクセルの切り取り
    pixel_value = frame[266:296, 438] # x座標:438 y座標:266~296
    
    # 各ピクセルの平均
    pixel_average = np.average(pixel_value, axis=1)
    
    try:
        result = np.where(pixel_average > 65)
        radius = result[0][-1]
        print(radius)

    except:
        print(e)

