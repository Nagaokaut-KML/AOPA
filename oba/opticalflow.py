import numpy as np
import cv2
# import pandas as pd

# video = input('video name:')
cap = cv2.VideoCapture('094448.mp4')

# Lucas-Kanade法のパラメータ
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# ランダムに色を100個生成（値0～255の範囲で100行3列のランダムなndarrayを生成）
color = np.random.randint(0, 255, (10000, 3))

# 最初のフレームの処理
end_flag, frame = cap.read()
gray_prev = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# 輪転機周りの座標を得る
ary = ([[[399,350]]])
np_ary = np.array(ary,dtype='float32')
print(np_ary)
feature_prev = np_ary
mask = np.zeros_like(frame)

result = [["prev_x","prev_y","next_x","next_y","diff_x","diff_y","move"]]

while(end_flag):
    # グレースケールに変換
    gray_next = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # オプティカルフロー検出
    feature_next, status, err = cv2.calcOpticalFlowPyrLK(gray_prev, gray_next, feature_prev, None, **lk_params)

    # オプティカルフローを検出した特徴点を選別（0：検出せず、1：検出した）
    good_prev = feature_prev[status == 1]
    good_next = feature_next[status == 1]

    # オプティカルフローを描画
    for i, (next_point, prev_point) in enumerate(zip(good_next, good_prev)):
        prev_x, prev_y = prev_point.ravel()
        next_x, next_y = next_point.ravel()
        mask = cv2.line(mask, (next_x, next_y), (prev_x, prev_y), color[i%100].tolist(), 2)
        frame = cv2.circle(frame, (next_x, next_y), 5, color[i].tolist(), -1)
        
        if (prev_x != next_x) and (prev_y != next_y):
            move = 1
        elif (prev_x != next_x):
            move = 1
        elif (prev_y != next_y):
            move = 1
        else:
            move = 0
    
    result.append([[prev_x,prev_y,next_x,next_y,(next_x-prev_x),(next_y-prev_y),move]])
    img = cv2.add(frame, mask)
    
    # ウィンドウに表示
    cv2.imshow('video', img)

    # qを押して停止
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

    # 次のフレーム、ポイントの準備
    gray_prev = gray_next.copy()
    feature_prev = good_next.reshape(-1, 1, 2)
    end_flag, frame = cap.read()

    # ファイル出力
    f = open('list_result_opticalflow_090921.csv', 'w')
    for x in result:
        f.write(str(x) + '\n')
    f.close


# 終了処理
cv2.destroyAllWindows()
cap.release()
