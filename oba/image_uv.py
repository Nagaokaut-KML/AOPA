import cv2
import numpy as np

video_name = input('video name (without .mp4) :') 
cap = cv2.VideoCapture(video_name+'.mp4')

list_result = [["no","bgr_mean","0or1or2","open/close/error"]]
count = 0
while(cap.isOpened()):
    end_flag, frame = cap.read()
    #cv2.imshow('video',frame)

    # UVの部分のみを切り取る
    img_cut = frame[90 : 110, 350 : 380]
    height, width, ch = img_cut.shape

    # BGR
    blue = np.array(img_cut[:,:,0])
    green = np.array(img_cut[:,:,1])
    red = np.array(img_cut[:,:,2])

    bgr = np.array([np.average(blue),np.average(green),np.average(red)])
    bgr_mean = np.mean(bgr)

    # UVが開いているかどうかの判定をする
    # open
    if bgr_mean >= 90 and bgr_mean <= 110:
        print (bgr_mean,'open')
        list_result.append([count,bgr_mean,1,"open"])
    # close
    elif bgr_mean < 90:
        print (bgr_mean,'close')
        list_result.append([count,bgr_mean,0,"close"])
    # 人がいる？
    else:
        print (bgr_mean,'error')
        list_result.append([count,bgr_mean,2,"error"])
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    count = count + 1

    f = open('list_result_image_uv_{video}.csv'.format(video=video_name), 'w')
    for x in list_result:
        f.write(str(x) + '\n')
    f.close

    cv2.imshow('img', frame)

    # qを押して停止
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()
