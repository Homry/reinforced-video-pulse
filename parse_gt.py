import os

import re
import datetime
import os

import numpy as np
import pandas

def match(file_path, csv):
    data = {}
    video_duration = 300
    listdir = os.listdir(f'{csv}/polar_output')
    a = (line for line in open(file_path, 'r', encoding='utf-8').read().split('\n\n'))
    for i in a:
        creation_line  = re.search(
            '([0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}-[0-9]{2}-[0-9]{2}).mp4', i)
        if creation_line is not None:
            video, video_creation_time = creation_line.group(0), datetime.datetime.strptime(creation_line.group(1), '%Y-%m-%d %H-%M-%S').timestamp()
            for pulse_file in listdir:
                pulse_ts = float(re.findall('[0-9]{10}.[0-9]{1,10}', pulse_file)[0])
                difference_ts = pulse_ts - video_creation_time
                if abs(difference_ts) < video_duration:
                    data[str(pulse_ts)] = video.split('.')[0]
                    print(video.split('.')[0], [pulse_ts])
    return data

ground = {
    "/media/sergey/hard_drive/part_1/room316/video/creation_time.txt": "/media/sergey/hard_drive/part_1/room316",
    "/media/sergey/hard_drive/part_1/room317/video/creation_time.txt": "/media/sergey/hard_drive/part_1/room317",
    "/media/sergey/hard_drive/part_2/room2103/video/creation_time.txt": "/media/sergey/hard_drive/part_2/room2103",
    "/media/sergey/hard_drive/part_2/room2115/video/creation_time.txt": "/media/sergey/hard_drive/part_2/room2115",
}
for key, val in ground.items():
    print(f'{key = }, {val = }')
    map_data = match(key)
    gt = os.listdir(f'{val}/polar_output')
    print(gt)
    for i in gt:
        print(i.split('_')[2].split('.csv')[0])
        file = i.split("_")[2].split(".csv")[0]
        data = pandas.read_csv(f'{val}/polar_output/pulsar_data_{file}.csv')
        os.mkdir(f'/home/sergey/reinforced-video-pulse/dataset_npy/gt/{map_data[str(file)]}')
        dataset = []
        for j in data.values:
            tmp = list(map(lambda x: int(x), j[1][1:-1].split(', ')))
            for j in tmp:
                dataset.append(j)
        begin = 0
        offset = 650
        end = 1300
        time = np.array([0, 10])
        while end <= len(dataset)-1:
            data = dataset[begin:end]
            begin += offset
            end += offset
            with open(f'/home/sergey/reinforced-video-pulse/dataset_npy/gt/{map_data[str(file)]}/{map_data[str(file)]}_{time[0]}-{time[1]}.npy', 'wb') as f:
                np.save(f, data)
            time += 5
