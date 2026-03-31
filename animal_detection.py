import cv2
import argparse
from ultralytics import YOLO

# Load model YOLO
model = YOLO("yolov8n.pt")

# Các class cần nhận diện + màu
TARGET_CLASSES = {
    "person": (255, 0, 0),   # xanh dương
    "dog": (0, 255, 0),      # xanh lá
    "cat": (0, 0, 255),      # đỏ
    "bird": (255, 255, 0),   # vàng (gà)
    "fish": (255, 0, 255)    # tím
}


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--conf", type=float, default=0.4)

    args = parser.parse_args()

    print("Loading YOLO model...")
    model = YOLO(args.model)

    cap = cv2.VideoCapture(args.input)

    if not cap.isOpened():
        print("Không mở được video")
        return

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        results = model(frame, conf=args.conf)[0]

        for box in results.boxes:

            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            conf = float(box.conf[0])

            if label not in TARGET_CLASSES:
                continue

            color = TARGET_CLASSES[label]

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Vẽ bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            text = f"{label} {conf:.2f}"

            cv2.putText(
                frame,
                text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        cv2.imshow("Animal Detection", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()