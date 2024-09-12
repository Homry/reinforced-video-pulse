# import re
# import datetime
# import os
# video_duration = 300
# listdir = os.listdir('./csv')
# a = (line for line in open('./creation_time.txt', 'r', encoding='utf-8').read().split('\n\n'))
# for i in a:
#     creation_line  = re.search(
#         '([0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}-[0-9]{2}-[0-9]{2}).mp4', i)
#     if creation_line is not None:
#         video, video_creation_time = creation_line.group(0), datetime.datetime.strptime(creation_line.group(1), '%Y-%m-%d %H-%M-%S').timestamp()
#         for pulse_file in listdir:
#             pulse_ts = float(re.findall('[0-9]{10}.[0-9]{1,10}', pulse_file)[0])
#             difference_ts = pulse_ts - video_creation_time
#             if abs(difference_ts) < video_duration:
#                 print(video, [pulse_ts])

import os
import pandas

data = []
# baseline = 'C:/Users/Homry/Documents/RnD/reinforced-video-pulse/dataset_npy'
baseline = '/home/sergey/reinforced-video-pulse/dataset_npy'
vals = os.listdir(f'{baseline}/value')
gt = os.listdir(f'{baseline}/transform')
for v in vals:
    if v in gt:
        gt_files = os.listdir(f'{baseline}/transform/{v}')
        for file in os.listdir(f'{baseline}/value/{v}'):
            if file in gt_files:
                data.append([f'{baseline}/value/{v}/{file}', f'{baseline}/transform/{v}/{file}'])


df = pandas.DataFrame(data)

df.to_csv('./dataset.csv', index=False)





