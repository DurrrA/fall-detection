from PyQt5 import QtWidgets, QtGui, QtCore
import sys
import cv2
from src.realtime.video_stream import VideoStream
from src.notifications.notifier import Notifier

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("Fall Detection System")
        self.setGeometry(100, 100, 800, 600)

        self.video_stream = VideoStream()
        self.notifier = Notifier()

        self.init_ui()
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def init_ui(self):
        self.label = QtWidgets.QLabel(self)
        self.label.setGeometry(10, 10, 640, 480)
        self.label.setScaledContents(True)

        self.result_label = QtWidgets.QLabel(self)
        self.result_label.setGeometry(10, 500, 640, 50)
        self.result_label.setFont(QtGui.QFont("Arial", 16))

        self.start_button = QtWidgets.QPushButton("Start Detection", self)
        self.start_button.setGeometry(670, 10, 120, 30)
        self.start_button.clicked.connect(self.start_detection)

        self.stop_button = QtWidgets.QPushButton("Stop Detection", self)
        self.stop_button.setGeometry(670, 50, 120, 30)
        self.stop_button.clicked.connect(self.stop_detection)

    def start_detection(self):
        self.video_stream.start()

    def stop_detection(self):
        self.video_stream.stop()

    def update_frame(self):
        frame = self.video_stream.get_frame()
        if frame is not None:
            # Here you would add the fall detection logic
            # For demonstration, we will just show the frame
            self.display_frame(frame)

    def display_frame(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        self.label.setPixmap(QtGui.QPixmap.fromImage(qt_image))

    def closeEvent(self, event):
        self.video_stream.stop()
        event.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())