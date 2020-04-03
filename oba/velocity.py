import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

class Velocity:
    def __init__(self, video_path):
       self.video_path = video_path
       cap = cv2.VideoCapture(video_path)


    def get_bgr_mean(self, frame):
        img_cut = frame[285 : 296, 435 : 445]
        blue = np.array(img_cut[:,:,0])
        green = np.array(img_cut[:,:,1])
        red = np.array(img_cut[:,:,2])
        bgr = np.array([np.average(blue),np.average(green),np.average(red)])
        bgr_mean = np.mean(bgr)
        return bgr_mean

    def get_mode(self, amplitude):
        # 0:機械停止中
        if amplitude <= 10:
            return 'stop'
            # 1:機械動作中
        elif amplitude >= 10 and amplitude <= 50:
            return 'move'
            # 2:エラー(人がかぶってたりする)
        else:
            return 'error'

    def get_amplitude(self, bgr_list, window):
        result_list = []
        for i in range(0, len(bgr_list)-window, window):
            y = bgr_list[i:i+window]
            result = max(y) - min(y)
            result_list.append(result)
            print (self.get_mode(result))
        return result_list

    def manage(self):
        cap = cv2.VideoCapture(self.video_path)
        count = 0
        count_list = []
        bgr_list = []
        amplitude = []
        while (cap.isOpened()):
            end_flag, frame = cap.read()
            if frame is None:
                break

            count += 1
            count_list.append(count)
            bgr_mean = self.get_bgr_mean(frame)
            bgr_list.append(bgr_mean)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        amplitude = self.get_amplitude (bgr_list=bgr_list, window=15)

        x = list(range(len(amplitude)))
        y = amplitude

        return x, y, bgr_list 

test = Velocity('./094448.mp4')
test.manage()