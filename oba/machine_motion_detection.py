import cv2
import numpy as np
import csv
import pandas as pd
import os

class DetectionMotion:

    def __init__(self, video_path):
        self.video_path = video_path
        
            
    # -- file path --
    def delete_file(self):
        if os.path.exists('./result/list_result_{}.csv'.format(self.video_path)):
            os.remove('./result/list_result_{}.csv'.format(self.video_path))
            print ('delete')

    # -- uv(BGR) -- あとは結果を出力するためのリストを整備する
    def detection_uv_bgr(self, frame):
        # test
        # print('detection_uv start')

        # UVの部分を切り取る
        img_cut = frame[90 : 110, 350 : 380]
        height, width, ch = img_cut.shape

        # BGR
        blue = np.array(img_cut[:,:,0])
        green = np.array(img_cut[:,:,1])
        red = np.array(img_cut[:,:,2])
        bgr = np.array([np.average(blue),np.average(green),np.average(red)])
        bgr_mean = np.mean(bgr)

        # UVが開いているかを判定する
        # open
        if bgr_mean >= 90 and bgr_mean <= 110:
            print('open')
            return 1
        # close
        elif bgr_mean < 90:
            print('close')
            return 0
        # other
        else:
            print('err')
            return 2    

    # -- paper -- あとは結果を出力するためのリストを整備する 
    def detection_paper(self, frame):
        # ピクセルの切り取り
        pixel_value = frame[266:296, 438] # x座標:438, y座標:266~296

        # 各ピクセルの平均
        pixel_average = np.average(pixel_value, axis=1)
        result = 0
        max_result = 0
        init_value = 265
        max_value = 265 # 初期値

        try:
            result = np.where(pixel_average > 65) # 各ピクセルで平均が65以上のものを抽出
            max_result = result[0][-1] + 266
            if abs(max_result - max_value) < 3 and max_value < max_result:
                max_value = max_result

            print (max_value)
            return max_value - init_value   

        except Exception as e:
            print(e)    
        
    def manage(self):
        result = []
        list_result = []
        count = 0

        print (self.video_path)
        self.delete_file()

        with open('./result/list_result_{}.csv'.format(self.video_path), 'a') as f:
            writer = csv.writer(f)
            init_result = ['video', 'frame', 'uv', 'paper_max_value']
            writer.writerow(init_result)
            
            cap = cv2.VideoCapture(self.video_path)
            video_name = self.video_path
            
            while(cap.isOpened()):
                print(count)
                end_flag, frame = cap.read()
                # -- uv --
                result_uv = self.detection_uv_bgr(frame)
                # -- paper --
                result_paper = self.detection_paper(frame)
                result = [video_name.rstrip(".mp4"), count, result_uv, result_paper]
                writer.writerow(result) 
                list_result.clear()
                count += 1

        cap.release()
        cv2.destroyAllWindows()
        

# -- 出力テスト--
# test = DetectionMotion('161439.mp4')
# test = DetectionMotion('161905.mp4')
test = DetectionMotion('085604.mp4')
# test.detection_uv()
test.manage()

