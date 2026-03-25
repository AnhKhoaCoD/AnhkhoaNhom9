"""
Vehicle Traffic Counter
=======================
Đếm lưu lượng xe trên đường từ video đầu vào.
Sử dụng: python vehicle_counter.py --input video.mp4 --output output.mp4

Yêu cầu:
    pip install opencv-python ultralytics numpy
"""

import cv2
import numpy as np
import argparse
from ultralytics import YOLO
from collections import defaultdict

# ─── Cấu hình ────────────────────────────────────────────────────────────────

# Các lớp xe cần đếm (theo COCO dataset)
VEHICLE_CLASSES = {
    2:  "Car",
    3:  "Motorcycle",
    5:  "Bus",
    7:  "Truck",
}

# Màu bounding box cho từng loại xe (BGR)
CLASS_COLORS = {
    2:  (0, 255, 0),      # Car      → xanh lá
    3:  (0, 165, 255),    # Motorcycle → cam
    5:  (255, 0, 0),      # Bus      → xanh dương
    7:  (0, 0, 255),      # Truck    → đỏ
}

# ─── Tracker đơn giản bằng centroid ──────────────────────────────────────────

class CentroidTracker:
    """
    Theo dõi xe bằng cách tính khoảng cách centroid giữa các frame.
    Mỗi xe được gán một ID duy nhất để tránh đếm trùng.
    """

    def __init__(self, max_disappeared=30, max_distance=80):
        self.next_id = 0
        self.objects = {}          # id → centroid
        self.disappeared = {}      # id → số frame mất tích
        self.class_map = {}        # id → class_id
        self.counted_ids = set()   # tập ID đã qua đường đếm

        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid, class_id):
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.class_map[self.next_id] = class_id
        self.next_id += 1

    def deregister(self, obj_id):
        del self.objects[obj_id]
        del self.disappeared[obj_id]
        del self.class_map[obj_id]

    def update(self, detections):
        """
        detections: list of (centroid_x, centroid_y, class_id)
        Trả về dict: id → (centroid, class_id)
        """
        if len(detections) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)
            return self._result()

        input_centroids = np.array([[d[0], d[1]] for d in detections])
        input_classes   = [d[2] for d in detections]

        # Chưa có object nào → đăng ký tất cả
        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i], input_classes[i])
            return self._result()

        # Tính ma trận khoảng cách Euclidean
        obj_ids       = list(self.objects.keys())
        obj_centroids = np.array(list(self.objects.values()))

        D = np.linalg.norm(obj_centroids[:, np.newaxis] - input_centroids, axis=2)

        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows, used_cols = set(), set()

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if D[row, col] > self.max_distance:
                continue

            obj_id = obj_ids[row]
            self.objects[obj_id]     = input_centroids[col]
            self.class_map[obj_id]   = input_classes[col]
            self.disappeared[obj_id] = 0
            used_rows.add(row)
            used_cols.add(col)

        unused_rows = set(range(D.shape[0])) - used_rows
        unused_cols = set(range(D.shape[1])) - used_cols

        for row in unused_rows:
            obj_id = obj_ids[row]
            self.disappeared[obj_id] += 1
            if self.disappeared[obj_id] > self.max_disappeared:
                self.deregister(obj_id)

        for col in unused_cols:
            self.register(input_centroids[col], input_classes[col])

        return self._result()

    def _result(self):
        return {oid: (self.objects[oid], self.class_map[oid])
                for oid in self.objects}


# ─── Hàm vẽ HUD thông tin ────────────────────────────────────────────────────

def draw_hud(frame, counts_by_class, total, fps):
    """Vẽ bảng thống kê và FPS lên góc trái trên."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    panel_w, panel_h = 240, 40 + 30 * (len(VEHICLE_CLASSES) + 1)
    cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h),
                  (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    y = 65
    for cls_id, name in VEHICLE_CLASSES.items():
        color = CLASS_COLORS[cls_id]
        count = counts_by_class.get(cls_id, 0)
        cv2.putText(frame, f"{name}: {count}", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y += 30

    cv2.putText(frame, f"TOTAL: {total}", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)


# ─── Chương trình chính ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Vehicle Traffic Counter")
    parser.add_argument("--input",      required=True,       help="Đường dẫn video đầu vào")
    parser.add_argument("--output",     default="output.mp4",help="Đường dẫn video đầu ra")
    parser.add_argument("--model",      default="yolov8n.pt", help="Model YOLOv8 (n/s/m/l/x)")
    parser.add_argument("--conf",       type=float, default=0.4,  help="Ngưỡng confidence (0-1)")
    parser.add_argument("--line-ratio", type=float, default=0.5,  help="Vị trí đường đếm (0-1)")
    parser.add_argument("--no-display", action="store_true", help="Không hiện cửa sổ preview")
    args = parser.parse_args()

    # Mở video
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise FileNotFoundError(f"Không mở được video: {args.input}")

    W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS_SRC = cap.get(cv2.CAP_PROP_FPS) or 30
    TOTAL_FRAMES = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Đường đếm nằm ngang
    COUNT_LINE_Y = int(H * args.line_ratio)

    # VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output, fourcc, FPS_SRC, (W, H))

    # Load model
    print(f"[INFO] Đang tải model {args.model} ...")
    model = YOLO(args.model)

    tracker = CentroidTracker(max_disappeared=25, max_distance=80)
    counts_by_class = defaultdict(int)
    total_count = 0

    frame_idx   = 0
    fps_display = 0.0
    tick = cv2.getTickCount()

    print(f"[INFO] Bắt đầu xử lý: {TOTAL_FRAMES} frames | {W}x{H} @ {FPS_SRC:.1f} fps")
    print(f"[INFO] Đường đếm tại y = {COUNT_LINE_Y}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # ── Detect ──────────────────────────────────────────────────────────
        results = model(frame, conf=args.conf, verbose=False)[0]

        detections = []   # (cx, cy, class_id, x1, y1, x2, y2)

        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in VEHICLE_CLASSES:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            detections.append((cx, cy, cls_id, x1, y1, x2, y2))

        # ── Track ───────────────────────────────────────────────────────────
        track_input = [(d[0], d[1], d[2]) for d in detections]
        tracked = tracker.update(track_input)

        # ── Đường đếm nằm ngang ─────────────────────────────────────────────
        for obj_id, (centroid, cls_id) in tracked.items():
            cx, cy = int(centroid[0]), int(centroid[1])

            # Xe vừa vượt qua đường đếm và chưa được ghi nhận
            if cy >= COUNT_LINE_Y and obj_id not in tracker.counted_ids:
                tracker.counted_ids.add(obj_id)
                counts_by_class[cls_id] += 1
                total_count += 1

        # ── Vẽ bounding box ─────────────────────────────────────────────────
        for det in detections:
            cx, cy, cls_id, x1, y1, x2, y2 = det
            color = CLASS_COLORS.get(cls_id, (128, 128, 128))
            label = VEHICLE_CLASSES[cls_id]

            # Tìm ID của object gần centroid này nhất
            obj_id_label = "?"
            for oid, (c, _) in tracked.items():
                if abs(int(c[0]) - cx) < 5 and abs(int(c[1]) - cy) < 5:
                    obj_id_label = str(oid)
                    break

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            tag = f"{label} #{obj_id_label}"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, tag, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.circle(frame, (cx, cy), 4, color, -1)

        # ── Vẽ đường đếm ────────────────────────────────────────────────────
        cv2.line(frame, (0, COUNT_LINE_Y), (W, COUNT_LINE_Y), (0, 255, 255), 2)
        cv2.putText(frame, "COUNTING LINE", (10, COUNT_LINE_Y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

        # ── HUD ─────────────────────────────────────────────────────────────
        # Tính FPS hiển thị mỗi 30 frame
        if frame_idx % 30 == 0:
            tock = cv2.getTickCount()
            fps_display = 30 / ((tock - tick) / cv2.getTickFrequency())
            tick = tock

        draw_hud(frame, counts_by_class, total_count, fps_display)

        # Progress
        if frame_idx % 100 == 0:
            pct = frame_idx / TOTAL_FRAMES * 100 if TOTAL_FRAMES else 0
            print(f"  Frame {frame_idx}/{TOTAL_FRAMES} ({pct:.1f}%) | "
                  f"Tổng xe: {total_count}")

        out.write(frame)

        if not args.no_display:
            cv2.imshow("Vehicle Counter (Q để thoát)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[INFO] Người dùng dừng sớm.")
                break

    # ── Kết thúc ────────────────────────────────────────────────────────────
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("\n" + "=" * 45)
    print("  KẾT QUẢ ĐẾM XE")
    print("=" * 45)
    for cls_id, name in VEHICLE_CLASSES.items():
        print(f"  {name:<15}: {counts_by_class.get(cls_id, 0):>5} xe")
    print("-" * 45)
    print(f"  {'TỔNG CỘNG':<15}: {total_count:>5} xe")
    print("=" * 45)
    print(f"\n[INFO] Video đã lưu tại: {args.output}")


if __name__ == "__main__":
    main()
