from src import FutureParseDetector, MediapipeDetector, VideoReader


if __name__ == '__main__':
    parser = FutureParseDetector(file_path='./videos/out2_amp5.mp4')
    parser.init_detector()
    parser.process()