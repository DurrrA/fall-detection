import cv2
import numpy as np

def preprocess_frame(frame):
    # Resize the frame to the required input size for the model
    frame_resized = cv2.resize(frame, (224, 224))  # Example size, adjust as needed
    # Normalize the pixel values to [0, 1]
    frame_normalized = frame_resized / 255.0
    # Convert the frame to a numpy array and add a batch dimension
    frame_array = np.expand_dims(frame_normalized, axis=0)
    return frame_array

def preprocess_video(video_path):
    # Capture video from the specified path
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Preprocess each frame
        processed_frame = preprocess_frame(frame)
        frames.append(processed_frame)
    
    cap.release()
    return np.vstack(frames)  # Stack frames into a single array

def preprocess_images(image_folder):
    import os
    images = []
    for filename in os.listdir(image_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(image_folder, filename)
            img = cv2.imread(img_path)
            processed_img = preprocess_frame(img)
            images.append(processed_img)
    return np.vstack(images)  # Stack images into a single array