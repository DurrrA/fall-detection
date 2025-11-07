import cv2
import numpy as np
from src.ml.model import load_model
from src.utils.preprocessing import preprocess_frame
from src.notifications.notifier import Notifier

class VideoStream:
    def __init__(self, model_path, notifier):
        self.model = load_model(model_path)
        self.notifier = notifier
        self.capture = cv2.VideoCapture(0)  # Use 0 for webcam or provide a video file path

    def start_stream(self):
        while True:
            ret, frame = self.capture.read()
            if not ret:
                break

            processed_frame = preprocess_frame(frame)
            prediction = self.model.predict(processed_frame)

            if self.is_fall_detected(prediction):
                self.notifier.send_alert("Fall detected!")
                self.display_alert(frame)

            self.display_frame(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.capture.release()
        cv2.destroyAllWindows()

    def is_fall_detected(self, prediction):
        # Assuming the model outputs a probability for fall detection
        return prediction[0] > 0.5  # Adjust threshold as necessary

    def display_frame(self, frame):
        cv2.imshow("Video Stream", frame)

    def display_alert(self, frame):
        cv2.putText(frame, "Fall Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Video Stream", frame)