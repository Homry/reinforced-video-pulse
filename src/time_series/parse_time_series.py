import numpy as np
from .time_series import TimeSeries


class ParseTimeSeries(TimeSeries):
    def __init__(self, file_name: str, current_save_dir: str, debug: bool = False):
        super().__init__(debug)
        self.__current_save_dir = current_save_dir
        self.__current_file_name = file_name.split('/')[-1].split('.')[0]
        self.__current_window = np.array([0, 10])
        self.__status = True
        self.__current_item = 0

    def update(self, file_name: str):
        self.__current_file_name = file_name
        self._vector = None
        self.__current_window = np.array([0, 10])
        self.__status = True
        self.__current_item = 0

    def add_in_vector(self, vector, status=None):
        if self._vector is None:
            super().init_vector(vector)
        else:
            super().add_in_vector(vector)
        self.__current_item += 1
        if self.__current_item == 300:
            self.__processing()

    def set_status(self):
        self.__status = False
        self.__current_item += 1

    def __processing(self):
        if self.__status:
            current_data = self._vector.copy()
            self.interpolate_signal()
            self.butter_filter()
            pca_data = self.pca(self._vector)
            self.__save_data(pca_data)
            self.__current_item = self.__current_item // 2
            window_offset = 5
            self._vector = [i[150::] for i in current_data]
        else:
            self.__current_item = 0
            window_offset = 10
            self.__status = True
            self._vector = None
        self.__current_window = self.__current_window + window_offset

    def __save_data(self, data):
        with open(f'{self.__current_save_dir}/{self.__current_file_name}_{self.__current_window[0]}-'
                  f'{self.__current_window[1]}.npy', 'wb') as f:
            np.save(f, np.array(data))
