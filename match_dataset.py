import re
import datetime
import os
video_duration = 300
listdir = os.listdir('./csv')
a = (line for line in open('./creation_time.txt', 'r', encoding='utf-8').read().split('\n\n'))
for i in a:
    creation_line  = re.search(
        '([0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}-[0-9]{2}-[0-9]{2}).mp4', i)
    if creation_line is not None:
        video, video_creation_time = creation_line.group(0), datetime.datetime.strptime(creation_line.group(1), '%Y-%m-%d %H-%M-%S').timestamp()
        for pulse_file in listdir:
            pulse_ts = float(re.findall('[0-9]{10}.[0-9]{1,10}', pulse_file)[0])
            difference_ts = pulse_ts - video_creation_time
            if abs(difference_ts) < video_duration:
                print(video, [pulse_ts])



# 2022-09-10 09-22-51.mp4 [1662790981.4648633]