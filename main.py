from src import FaceDetector, TimeSeries

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    forehead_time_series = TimeSeries()
    mouth_time_series = TimeSeries()
    face = FaceDetector('./videos/test_amp5.mp4', forehead_time_series, mouth_time_series)
    face.test_process()
    # time_series.distance_filter()
    # time_series.interpolate_signal()


    print(time_series)
