from gui.main_window import MainWindow
from realtime.video_stream import VideoStream
import sys
from PyQt5.QtWidgets import QApplication

def main():
    app = QApplication(sys.argv)
    video_stream = VideoStream()
    main_window = MainWindow(video_stream)
    main_window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()