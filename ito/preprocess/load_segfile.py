import numpy as np
import os
import csv
from glob import glob
from sklearn.utils import shuffle
from tqdm import tqdm
import pandas as pd
import json
import re

class LoadSegFile:
    def __init__(self, length=400):
        self.time_pd = ""
        self.seg_file = {}
        self.time_pd = pd.read_csv('data/20180903time.csv')
        self.load_segfile('03')
        self.time_pd = pd.read_csv('data/20180904time.csv')
        self.load_segfile('04')

        fw = open('./data/segFile.json','w')
        json.dump(self.seg_file,fw,indent=4)

    def load_segfile(self, date):
        """
            dateの時間の画像pathファイルの読み込み

            Parameters
            ----------
            date : string
                何日か(04)
        """
        with open('./data/09' + date + 'seg_image_path.csv') as f:
            reader = csv.reader(f) 
            lines = [line for line in f]
            currentRow = 1

            for row in tqdm(csv.reader(lines), total=len(lines)):
                time = self.get_time(row[0],date)
                self.seg_file.setdefault(time, [])
                self.seg_file[time].append(row[0])
                currentRow += 1
            

    def get_time(self, image_path, date):
        """
            image_pathの画像が何時何分のやつなのかを取得する

            Parameters
            ----------
            image_path : string
                対象となる画像のパス
            date : string
                日付(03)
                
            Returns
            -------
            time : string
                画像の時間 ('03072732')
        """
        target_data = self.time_pd[self.time_pd['path'] == image_path]
        try:
            time = str(target_data['hour'].values[0])+':' + str(target_data['minute'].values[0]) + ':' +str(target_data['second'].values[0])
            str_time = '2018-01-' + date +' ' + time
            time_pd = pd.to_datetime(str_time)
            return time_pd.strftime("%d%H%M%S") 
        except Exception as e:
            print(e)
            print('[ERROR]',image_path, date)
            print('[ERROR]',target_data)

loadSegFile = LoadSegFile()