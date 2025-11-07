import os

class Config:
    # Model parameters
    MODEL_PATH = os.path.join('models', 'fall_detection_model.h5')
    INPUT_SHAPE = (224, 224, 3)  # Example input shape for the model
    NUM_CLASSES = 2  # Fall and Not-Fall

    # Notification settings
    EMAIL_NOTIFICATIONS_ENABLED = True
    SMS_NOTIFICATIONS_ENABLED = False
    NOTIFICATION_RECIPIENTS = {
        'email': 'recipient@example.com',
        'phone': '+1234567890'
    }

    # Paths to resources
    DATASET_PATH = os.path.join('data', 'fall_dataset')
    CSV_FILE_PATH = os.path.join(DATASET_PATH, 'data.csv')
    FALL_IMAGES_PATH = os.path.join(DATASET_PATH, 'images', 'fall')
    NOT_FALL_IMAGES_PATH = os.path.join(DATASET_PATH, 'images', 'not-fall')

    # Real-time video settings
    VIDEO_SOURCE = 0  # 0 for webcam, or provide a video file path
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480

    # Other settings
    CONFIDENCE_THRESHOLD = 0.5  # Threshold for fall detection confidence
    NOTIFICATION_DELAY = 5  # Delay in seconds between notifications