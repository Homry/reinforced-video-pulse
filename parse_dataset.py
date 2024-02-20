from src import FutureParseDetector, MediapipeDetector, VideoReader



if __name__ == '__main__':
    parser = FutureParseDetector(file_path='D:/датасет/room2115/video/2022-09-16 09-50-35.mp4')
    parser.init_detector()
    parser.process()
    #"D:\датасет\room2115\video\2022-09-16 09-50-35.mp4"
