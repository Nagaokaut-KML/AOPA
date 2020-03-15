import cv2
import numpy as np
import csv
import pandas as pd
import os

class DetectionMotion:
    def __init__(self, video_path):
        self.video_path = video_path
        self.source = video_path.lstrip('./video/')
        
    # -- file path --
    def delete_file(self):
        print('source',self.source)
        if os.path.exists('./result/list_result_{}.csv'.format(self.source)):
            os.remove('./result/list_result_{}.csv'.format(self.source))
            print ('delete')

    # -- uv(BGR) -- 
    def detection_uv_bgr(self, frame):
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

    # -- paper --  
    def detection_paper(self, frame):
        # ピクセルの切り取り
        pixel_value = frame[266:296, 438] # x座標:438, y座標:266~296

        # 各ピクセルの平均
        pixel_average = np.average(pixel_value, axis=1)

        try:
            result = np.where(pixel_average > 65) # 各ピクセルで平均が65以上のものを抽出
            print('result', result)
            radius = result[0][-1]
            return radius
        except Exception as e:
            print(e)    
        
    def manage(self):
        result = []
        list_result = []
        count = 0
        print (self.video_path) # 確認用
        self.delete_file()

        with open('./result/list_result_{}.csv'.format(self.source), 'a') as f:
            writer = csv.writer(f)
            init_result = ['video', 'frame', 'uv', 'paper_max_value']
            writer.writerow(init_result)
            
            # 動画読み込み
            cap = cv2.VideoCapture(self.video_path)
            video_name = self.source
            
            while(cap.isOpened()):
                print(count)
                end_flag, frame = cap.read()
                # -- uv --
                result_uv = self.detection_uv_bgr(frame)
                # -- paper --
                result_paper = self.detection_paper(frame)
                # 結果出力
                result = [video_name.rstrip(".mp4"), count, result_uv, result_paper]
                writer.writerow(result) 
                list_result.clear()
                count += 1

        cap.release()
        cv2.destroyAllWindows()

# -- 出力テスト--最終的には消す
test2 = DetectionMotion('video/161905.mp4')
test2.manage()
