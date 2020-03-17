import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import pyplot
from PIL import Image

#画像の読み込み
im = Image.open("./094448/_0969.jpg")

#画像をarrayに変換
im_list = np.asarray(im)
#貼り付け
plt.imshow(im_list)
#表示
# plt.show()

cap = cv2.VideoCapture('./video/094448.mp4')
count = 0
count_list = []
pixel_value_mean_list = []

while(cap.isOpened()):
    end_flag, frame = cap.read()

    # 回転部分の切り取り
    pixel_value = frame[289, 438]
    pixel_value_mean = np.average(pixel_value)
    
    print (pixel_value,':', pixel_value_mean)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    count_list.append(count)
    pixel_value_mean_list.append(pixel_value_mean)
 
x = count_list
y = pixel_value_mean_list
pyplot.plot(x, y)
pyplot.show()

cap.release()
cv2.destroyAllWindows()

