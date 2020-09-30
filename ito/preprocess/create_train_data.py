import numpy as np
import os
import csv
from glob import glob
from sklearn.utils import shuffle
import pandas as pd
import json
import re

class CreateTrainData:
    def __init__(self, length=400):
        self.data = {}
        self.time_pd = ""
        self.seg_file = {}
        with open('./data/0903formated_label.csv') as f:
            self.time_pd = pd.read_csv('data/20180903time.csv')
            self.load_segfile('03')
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                time = self.format_time(row[0], '03')
                self.data[time] = {'label': int(row[1]), 'date':row[0]}
        
                    
        with open('./data/0904formated_label.csv') as f:
            self.time_pd = pd.read_csv('data/20180904time.csv')
            self.load_segfile('04')
            reader = csv.reader(f)
            header = next(reader)  
            for row in reader:
                time = self.format_time(row[0], '04')
                self.data[time] = {'label': row[1], 'date':row[0]}

        fw = open('./data/trainData.json','w')
        json.dump(self.data,fw,indent=4)

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
            for row in reader:
                time = self.get_time(row[0],date)
                self.seg_file.setdefault(time, [])
                self.seg_file[time].append(row[0])

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
        time = target_data.iat[0,1] +':' + str(target_data.iat[0,2]) + ':' +str(target_data.iat[0,3])
            str_time = '2018-01-' + date +' ' + time
            pd_time = pd.to_datetime(time_j)
        return pd_time.strftime("%d%H%M%S") 

    def format_time(seld, time, day):
        """
            timeをkeyの形に変形する

            Parameters
            ----------
            time : string
                7:5:14
            day : string
                日付(03)
                
            Returns
            -------
            format : string
                画像の時間 ('03072732')
        """
        str_time = '2018-01-' + day +' ' + time
        pd_time = pd.to_datetime(time_j)
        return pd_time.strftime("%d%H%M%S") 

createTrainData = CreateTrainData()