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