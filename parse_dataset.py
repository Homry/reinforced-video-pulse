from src import FutureParseDetector, MediapipeDetector, VideoReader



if __name__ == '__main__':
    parser = FutureParseDetector(file_path='D:/датасет/room2115/video/2022-09-10 13-13-58.mp4')
    parser.init_detector()
    parser.process()
