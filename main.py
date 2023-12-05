import time
import tqdm
from src import FaceFeaturePointsDetector, TimeSeries

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # video_time = [41, 45, 69, 70]
    handler = ['media']
    video_time = [41, 43, 45, 50 ,55 ,60, 70, 75]
    # video_time = [41, 45, 69, 70]
    res = []
    for i in handler:
        for j in video_time:
            res.append((i, j ))
    total = len(res)
    for number, i in enumerate(res):
        print(f'{i}, {number+1}/{total}')

        dir_ = f'npy_data/2022-09-10 09-22-51/{i[1]}'
        time_ = time.time()
        time_series = TimeSeries(debug=False)
        # face = FaceFeaturePointsDetector('./videos/2022-09-10 09-22-51.mp4', time_series, handler=i[0], debug=False, time_=i[1]* 60 * 30)
        face = FaceFeaturePointsDetector('D:/датасет/room2115/video/2022-09-10 13-13-58.mp4', time_series, handler=i[0],
                                         debug=False,
                                         time_=i[1] * 60 * 30)
        face.init_detector()
        face.process_video()
        # time_series.distance_filter()

        time_series.interpolate_signal()
        time_series.butter_filter()
        data = time_series.slice_vector(10)
        for signal in data:
            pca_signal = time_series.pca(signal)
            # smooth_signal = time_series.polynomial_smoothing(pca_signal)
            # time_series.find_signals_peaks(smooth_signal, pca_signal)
            # time_series.find_signals_peaks(pca_signal, pca_signal)
            time_series.correlation_after_pca(pca_signal, dir_, f'{i[0]}')

        # time_series.create_pulse_wave(10, dir_, f'{i[0]}')
        time_series.plot()
        print(f'time = {time.time() - time_}')


# 2022-09-10 13-13-58.mp4 [1662804846.0179787]
