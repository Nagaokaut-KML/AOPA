import argparse
import os
import numpy as np
from tqdm import tqdm
import glob
import cv2
import csv
import json
import collections as cl
from PIL import Image
import pandas as pd

from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *
import yaml


def setting_model(model, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'])

    device = torch.device("cuda:0")
    model = model.to(device)
    model = nn.DataParallel(model)
    return model 


    
def check_direct(path):
    if not os.path.exists(path):
        os.mkdir(path)

def check_file(path):
    return os.path.exists(path)


def load_img(p):
    if( check_file(p)):
        mean=(0.485, 0.456, 0.406)
        std=(0.229, 0.224, 0.225)
        x =  np.array(Image.open(p).convert('RGB'), dtype=np.float32)
        x /= 255.0
        x -= mean
        x /= std
        x = x.transpose((2, 0, 1))
    else:
        priint('do not exit file', p)

    return x

def encode(x):
    x, low_level_feat = model.module.backbone(x)
    x = model.module.aspp(x) #in: torch.Size([10, 320, 30, 40]) out: torch.Size([10, 256, 30, 40])
    return x

if __name__ == '__main__':
    f = open("./params/deeplab.yaml", "r+")
    params = yaml.load(f)
    model = DeepLab(num_classes=5, backbone='mobilenet')
    model = setting_model(model, params["weight"])
    model.eval()
    data_dict = cl.OrderedDict()
    batch_size =0 
    device = 'cuda'
    last_index = 0

    for index, day in enumerate(params['dataloader']['days']):
        all_image_paths = sorted(glob.glob(params["dataloader"]["base_path"] + day + '*/*.jpg'))
        time_pd = pd.read_csv(params['dataloader']['time_path'][index])
        result03 = []
        result03_paths = []
        for image_path_index in tqdm(range(0, len(all_image_paths), 8)):
            image_path = all_image_paths[image_path_index]
            if day == '20180903/':
                target_time = time_pd[ time_pd['path'] == image_path.replace('/groups2/gcb50278/all_frames/', '')]
            else:
                target_time = time_pd[ time_pd['path'] == image_path.replace('/groups2/gcb50278/all_frames', '')]
            if len(target_time.index) != 0:
                if target_time.iloc[0]['hour'] > 6 and target_time.iloc[0]['hour'] < 19 :
                    time = str(target_time.iloc[0]['hour']).zfill(2) + str(target_time.iloc[0]['minute']).zfill(2) + str(target_time.iloc[0]['second']).zfill(2)

                    im = load_img(image_path)
                    x = torch.from_numpy(im).float().to(device)
                    x = x.view(1,3,480,640)
                    seg = encode(x)
                    seg = seg.view(256, 30, 40)
                    seg = seg.cpu().detach().numpy()
                    result03.append(seg)
                    result03_paths.append([image_path, time])
                    last_index = image_path_index

        with open( params["dataloader"]["csv"][index] + ".csv","w") as f:
            writer = csv.writer(f, lineterminator="\n") 
            writer.writerows(result03_paths)

        result03 = np.array(result03)
        np.savez(params["dataloader"]["save_npz"][index] + ".npz", result03)

        # result03 = []
        # result03_paths = []
        # for image_path_index in tqdm(range(last_index + 4,len(all_image_paths),4)):
        #     image_path = all_image_paths[image_path_index]
        #     if day == '20180903/':
        #         target_time = time_pd[ time_pd['path'] == image_path.replace('/groups2/gcb50278/all_frames/', '')]
        #     else:
        #         target_time = time_pd[ time_pd['path'] == image_path.replace('/groups2/gcb50278/all_frames', '')]
        #     if len(target_time.index) != 0:
        #         if target_time.iloc[0]['hour'] > 6 and target_time.iloc[0]['hour'] < 19 :
        #             time = str(target_time.iloc[0]['hour']).zfill(2) + str(target_time.iloc[0]['minute']).zfill(2) + str(target_time.iloc[0]['second']).zfill(2)
        #             im = load_img(image_path)
        #             x = torch.from_numpy(im).float().to(device)
        #             x = x.view(1,3,480,640)
        #             seg = encode(x)
        #             seg = seg.cpu().detach().numpy()
        #             result03.append(seg)
        #             result03_paths.append([image_path, time])

        # with open( params["dataloader"]["csv"][index] + "_02.csv","w") as f:
        #     writer = csv.writer(f, lineterminator="\n") 
        #     writer.writerows(result03_paths)

        # result03 = np.array(result03)
        # np.savez(params["dataloader"]["save_npz"][index] + "_02.npz", result03)

    print('finish')


