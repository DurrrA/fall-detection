def draw_bounding_box(frame, bbox, label, color=(0, 255, 0), thickness=2):
    x, y, w, h = bbox
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
    return frame

def visualize_detections(frame, detections):
    for detection in detections:
        bbox = detection['bbox']
        label = detection['label']
        frame = draw_bounding_box(frame, bbox, label)
    return frame

def display_result(frame, result_text):
    cv2.putText(frame, result_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    return frame