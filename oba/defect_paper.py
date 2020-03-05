import numpy as np
import cv2
import os 
import csv

# 動画読み込み
VIDEO_SRC = input('video name')
cap = cv2.VideoCapture(VIDEO_SRC + '.mp4')

# 初期値
list_result = []
list_paper = [266]
max_result = 0
max_prev = 0
max_next = 0
count = 0
count_same_coordinate = 1
result_same_coordinate = 0
update = 0
update_count = 0
list_count_same_coodinate = []
move = 0
list_detect = [] # 350回に1回でもupdateがあるかを判定するためのlist
result = '' # 動いているかどうかの判定

while(cap.isOpened()):
    end_flag, frame = cap.read()
    
    for i in range(266, 297):
        pixel_value = frame[i, 438]

        if np.average(pixel_value) > 65:
            result = 1
        else:
            result = 0
    
        count += 1 

        if list_paper != [] and max_result < list_paper[-1]:
            max_result = list_paper[-1]
            max_next = max_result
        
        if max_prev != max_next:
            update = 1
            print ('update')
            max_prev = max_next
            result_same_coordinate = count_same_coordinate
            list_count_same_coodinate.append(count_same_coordinate)
            count_same_coordinate = 0
        else:
            update = 0

        if update == 1:
            update_count += 1
            if update_count < 350:
                print ('move')
            else:
                print ('dont move')
        else:
            print ('dont move')
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    
cap.release()
cv2.destroyAllWindows()
