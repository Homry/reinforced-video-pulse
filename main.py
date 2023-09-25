import time

from src import FaceFeaturePointsDetector, TimeSeries

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    time_ = time.time()
    time_series = TimeSeries()
    face = FaceFeaturePointsDetector('./videos/test_amp5.mp4', time_series)
    face.init_detector()
    face.process_video()
    # time_series.distance_filter()

    time_series.interpolate_signal()
    time_series.butter_filter()
    time_series.pca()
    time_series.find_signals_peaks()
    print(f'time = {time.time()-time_}')

