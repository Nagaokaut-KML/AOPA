import glob
from velocity import Velocity 
import matplotlib.pyplot as plt
import csv

def save_fig(file_name,save_name, x, y):
    plt.figure()
    plt.title(save_name)
    plt.ylim([0, 200])
    plt.plot(x, y)
    plt.savefig('./20180903_{}/{}.png'.format(save_name, file_name))

videos = glob.glob('./20180903/*')
list_result = []
output_list = [] 

with open('./bgr_list_20180903.csv', 'a') as f:
    writer = csv.writer(f)

    for i, file in enumerate(videos):
        file_path = file
        print (file_path)
        file_name = file_path[2:17]
        print (file_name)
        source = Velocity(file_path)
        x, y, bgr_list = source.manage()
        # save_fig(file_name[9:15], 'amplitude', x, y)
        # save_fig(file_name[9:15], 'wave', list(range(len(bgr_list))), bgr_list)
        list_result += y
        output_list.append(bgr_list)
        writer.writerow(output_list)
        output_list.clear()

# plt.figure()
# plt.hist(list_result, bins=256)
# plt.savefig('./histgram.png')