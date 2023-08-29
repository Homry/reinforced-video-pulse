from src import FaceDetector, TimeSeries

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    time_series = TimeSeries()
    face = FaceDetector('./videos/test_amp5.mp4', time_series)
    face.process_video()

    time_series.interpolate_signal()
    time_series.distance_filter()


    print(time_series)
