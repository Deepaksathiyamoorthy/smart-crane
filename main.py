import cv2
import numpy as np
from ultralytics import YOLO
import pyttsx3

# Allowed object classes
ALLOWED_LABELS = [
    'car', 'truck', 'bus', 'bicycle', 'motorcycle', 'person',
    'dog', 'cat', 'cow', 'horse', 'sheep', 'elephant', 'bear', 'zebra', 'giraffe'
]

# Approximate real widths in meters
KNOWN_WIDTHS = {
    'person': 0.5,
    'bicycle': 1.7,
    'car': 1.8,
    'truck': 2.5,
    'bus': 2.5,
    'motorcycle': 1.0,
    'dog': 0.6,
    'cat': 0.5,
    'cow': 1.7,
    'horse': 1.5,
    'sheep': 1.3,
    'elephant': 3.0,
    'bear': 2.0,
    'zebra': 2.0,
    'giraffe': 3.5,
}

COLLISION_THRESHOLD = 2.0  # meters
DEFAULT_FOCAL_LENGTH = 650

# Voice assistant
engine = pyttsx3.init()
engine.setProperty('rate', 160)
engine.setProperty('volume', 1.0)

def estimate_distance(pixel_width, known_width, focal_length):
    if pixel_width == 0:
        return None
    return (known_width * focal_length) / pixel_width

def main():
    # --- Open cameras ---
    front_cam = cv2.VideoCapture(0)
    back_cam = cv2.VideoCapture(1)
    cams = []

    if front_cam.isOpened():
        cams.append((front_cam, "Front"))
    else:
        print("Error: Front camera not found. Exiting.")
        return

    if back_cam.isOpened():
        cams.append((back_cam, "Back"))
    else:
        print("Back camera not found. Running only front camera.")

    # --- Load YOLO model ---
    model_path = "yolo11n.pt"
    model = YOLO(model_path)

    focal_length = DEFAULT_FOCAL_LENGTH
    risk_announced = False  # announce only once

    print("Starting detection. Press 'q' to quit.")

    while True:
        frames = []
        risk_objects_current = []

        for cap, label in cams:
            ret, frame = cap.read()
            if not ret:
                frame = 255 * np.ones((360, 480, 3), dtype=np.uint8)
                cv2.putText(frame, f"No {label} Camera", (50, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                frames.append(frame)
                continue

            results = model(frame, imgsz=320)[0]
            names = model.names

            for box in results.boxes:
                cls_id = int(box.cls[0])
                obj_label = names[cls_id]

                if obj_label not in ALLOWED_LABELS:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                pixel_width = x2 - x1
                distance = estimate_distance(pixel_width, KNOWN_WIDTHS.get(obj_label, 1.0), focal_length)

                # Box color and warning
                if distance and distance < COLLISION_THRESHOLD:
                    box_color = (0, 0, 255)
                    cv2.putText(frame, "COLLISION RISK!", (x1, y2 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    risk_objects_current.append(f"{obj_label} at {distance:.1f}m")
                else:
                    box_color = (0, 255, 0)

                # Draw box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(frame, f"{obj_label}: {distance:.2f}m" if distance else obj_label,
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.putText(frame, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            frames.append(cv2.resize(frame, (480, 360)))

        # Announce risk objects only once at start
        if risk_objects_current and not risk_announced:
            alert_text = "Warning! " + ", ".join(risk_objects_current)
            print("ALERT:", alert_text)
            engine.say(alert_text)
            engine.runAndWait()
            risk_announced = True

        # Combine frames horizontally if more than 1 camera
        display_frame = frames[0] if len(frames) == 1 else cv2.hconcat(frames)
        cv2.imshow("Crane Collision Avoidance", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- Cleanup ---
    for cap, _ in cams:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
