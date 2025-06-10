from ultralytics import YOLO
import cv2
import pandas as pd

model = YOLO("yolov8n.pt")
price_df = pd.read_csv("prices.csv")
price_dict = dict(zip(price_df["object"], price_df["price"]))

def run_detection():
    cap = cv2.VideoCapture(0)
    detected_items = {}

    added = False  # Flag to avoid scanning multiple items in one key press

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.6)[0]  # Increase confidence threshold

        frame_copy = frame.copy()
        detected_this_frame = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id].lower()

            if class_name not in price_dict:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = (255, 0, 0)  # Blue box

            # Track first valid object only
            if not detected_this_frame:
                detected_this_frame.append(class_name)

                # If user presses 's', add object and turn box green
                if cv2.waitKey(1) & 0xFF == ord('s') and not added:
                    detected_items[class_name] = detected_items.get(class_name, 0) + 1
                    color = (0, 255, 0)  # Green
                    added = True
                else:
                    added = False

                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame_copy, class_name, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            else:
                continue  # Ignore background items

        cv2.imshow("Scan Item (Press 's' to add, 'q' to finish)", frame_copy)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return detected_items
