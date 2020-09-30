import os

import tensorboardX as tbx
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataloaders.segment_dataloader import Segment_dataloader

from utils.loss import SegmentationLosses
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.metrics import Evaluator
from model.cnn_lstm import CNN_LSTM
# from model.mobilenet_LSTM import CNN_LSTM

def predict(date,loader):
    model.eval()
    evaluator.reset()
    tbar = tqdm(loader)
    test_loss = 0.0
    result = []
    for i, sample in enumerate(tbar):
        image, target, time = sample['image'], sample['label'], sample['time']
        if cuda:
            image, target = image.cuda(), target.cuda()
        with torch.no_grad():
            output = model(image)

        # loss = F.nll_loss(output, target)
        loss = criterion(output, target)
        test_loss += loss.item()
        
        pred = output.data.cpu().numpy()
        pred = np.argmax(pred, axis = 2)
        for batch, row in enumerate(pred):
            for index, ans in enumerate(row):
                result.append([time[batch][index], ans])


    result = np.array(result)
    np.savetxt('./result/result_' + str(date) + '.csv',result,delimiter=',')


def test(mode,loader, epoch):
    model.eval()
    evaluator.reset()
    tbar = tqdm(loader)
    test_loss = 0.0
    for i, sample in enumerate(tbar):
        image, target, time = sample['image'], sample['label'], sample['label']
        if cuda:
            image, target = image.cuda(), target.cuda()
        with torch.no_grad():
            output = model(image)

        # loss = F.nll_loss(output, target)
        loss = criterion(output, target)
        test_loss += loss.item()
        
        pred = output.data.cpu().numpy()
        target = target.cpu().numpy()
        # Batches are set one at a time to argmax because 3D arrays cannot be converted to 2D arrays with argmax.
        
        batch = pred.shape[0]
        reshape_pred = np.zeros((batch,200),dtype=int)
        for index in range(batch):
            reshape_pred[index] = np.argmax(pred[index], axis = 1)
        # Add batch sample into evaluator
        evaluator.add_batch(target, reshape_pred)

        tbar.set_description(mode + ' loss: %.7f' % (test_loss / (i + 1)))
    # Fast test during the training
    Acc = evaluator.Pixel_Accuracy()
    Acc_class = evaluator.Pixel_Accuracy_Class()
    mIoU = evaluator.Mean_Intersection_over_Union()
    FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
    writer.add_scalar(mode + "/loss", test_loss/(i + 1), epoch)
    writer.add_scalar(mode + "/acc", Acc, epoch)
    writer.add_scalar(mode + "/Acc_class", Acc_class, epoch)
    writer.add_scalar(mode + "/mIoU", mIoU, epoch)
    writer.add_scalar(mode + "/fwIoU", FWIoU, epoch)    

"""
    load model
"""
cuda = True
if cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    
model = CNN_LSTM(100,256, 64, 12)
model = model.to(device)
weight_path = "./model_weigth/cnn_lstm/weight_epoc11.pth"
checkpoint = torch.load(weight_path)
model.load_state_dict(checkpoint)

"""
    define parameter
"""
batch_size = 4
lr = 0.01
optimizer = torch.optim.Adam(model.parameters(),lr=lr)
criterion = SegmentationLosses(cuda=cuda).build_loss(mode="ce")
evaluator = Evaluator(12)


"""
    load dataloader and prediction
"""
trainset = Segment_dataloader(mode="train")
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
predict('0903', train_loader)
del trainset
del train_loader


valset = Segment_dataloader(mode="val")
val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False)
predict('0904', val_loader)
del valset
del val_loader

testset = Segment_dataloader(mode="test")
test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
scheduler = LR_Scheduler("poly", 0.001, 100, len(test_loader))
predict('0915', test_loader)

# writer = tbx.SummaryWriter("log-1")
exp = 1

print("start learning.")

# test('test', test_loader, epoch)

writer.close()
