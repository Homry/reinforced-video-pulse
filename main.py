import time

from src import FaceFeaturePointsDetector, TimeSeries

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    time_ = time.time()
    time_series = TimeSeries(debug=False)
    face = FaceFeaturePointsDetector('./videos/out2_amp5.mp4', time_series, debug=False)
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
        time_series.find_signals_peaks(pca_signal, pca_signal)

    time_series.create_pulse_wave(10)
    time_series.plot()
    print(f'time = {time.time()-time_}')

