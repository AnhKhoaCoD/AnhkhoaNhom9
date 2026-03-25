"""
╔══════════════════════════════════════════════════════════════════╗
║           VEHICLE COUNTER — 1 LANE (Đếm 1 làn)                 ║
╠══════════════════════════════════════════════════════════════════╣
║  Stack sử dụng:                                                  ║
║  - YOLOv8   : Nhận diện xe trong từng frame                     ║
║  - ByteTrack: Theo dõi từng xe qua nhiều frame (gán track ID)   ║
║  - OpenCV   : Đọc/ghi video, vẽ lên frame                       ║
╚══════════════════════════════════════════════════════════════════╝

Usage:
    python count_1lane.py --input video.mp4 --output output.mp4 --lane left --show
    python count_1lane.py --input video.mp4 --output output.mp4 --lane right --split 0.45
"""

import argparse
import cv2
import numpy as np
from collections import defaultdict, deque, Counter
from ultralytics import YOLO
import supervision as sv


# ═══════════════════════════════════════════════════════════════════
#  CÁC HẰNG SỐ CẤU HÌNH
# ═══════════════════════════════════════════════════════════════════

# YOLO được train trên dataset COCO gồm 80 class.
# Mỗi class có một ID số nguyên cố định.
# 2 = "car" (ô tô con), 7 = "truck" (xe tải)
VEHICLE_CLASSES   = {2: "Car", 7: "Truck"}
VEHICLE_CLASS_IDS = list(VEHICLE_CLASSES.keys())   # [2, 7]

# Màu BGR (Blue-Green-Red) dùng khi vẽ label lên frame.
# Lưu ý: OpenCV dùng BGR, không phải RGB như thông thường.
CLASS_BGR = {
    2: (0, 140, 255),    # Cam → Car
    7: (255, 80,  0),    # Xanh dương → Truck
}

# ── Tham số cho thuật toán Majority Voting (chống nhảy class) ────
# ByteTrack giữ track_id ổn định, nhưng YOLO đôi khi đổi class giữa
# các frame (frame này bảo là Car, frame sau bảo là Truck).
# Giải pháp: lưu 15 lần predict gần nhất → lấy class xuất hiện nhiều nhất.
CLASS_VOTE_WINDOW = 15   # số frame lưu lại để vote

# ── Tham số cho thuật toán Centroid Crossing (đếm xe qua vạch) ───
# Thay vì kiểm tra đúng 1 điểm, dùng một "vùng đệm" ±6px
# quanh vạch đếm. Mục đích: bắt được xe di chuyển nhanh mà có thể
# "nhảy qua" vạch trong 1 frame mà không dừng đúng tại vạch.
CROSS_BUFFER = 6   # pixel


# ═══════════════════════════════════════════════════════════════════
#  PARSE ARGUMENTS — Đọc tham số từ command line
# ═══════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description="Vehicle Counter — 1 Lane")
    p.add_argument("--input",  required=True,  help="Đường dẫn video đầu vào")
    p.add_argument("--output", required=True,  help="Đường dẫn video đầu ra")
    p.add_argument("--model",  default="yolov8n.pt",
                   help="File model YOLO. 'n'=nhỏ/nhanh, 's/m/l/x'=to/chính xác hơn")
    p.add_argument("--conf",   type=float, default=0.35,
                   help="Ngưỡng confidence (0.0-1.0). Thấp=bắt nhiều hơn, Cao=bắt ít hơn")
    p.add_argument("--iou",    type=float, default=0.35,
                   help="Ngưỡng IoU cho NMS. Thấp=tách được xe đứng gần nhau")
    p.add_argument("--line",   type=float, default=0.45,
                   help="Vị trí vạch đếm NGANG (0.0=trên cùng, 1.0=dưới cùng, 0.5=giữa)")
    p.add_argument("--split",  type=float, default=0.5,
                   help="Vị trí đường DỌC chia làn trái/phải (0.5=giữa frame)")
    p.add_argument("--lane",   choices=["left", "right"], default="left",
                   help="Làn cần đếm: 'left' hoặc 'right'")
    p.add_argument("--show",   action="store_true",
                   help="Hiển thị cửa sổ realtime. Nhấn Q để thoát")
    p.add_argument("--speed",  type=float, default=0.25,
                   help="Tốc độ video output (1.0=bình thường, 0.5=chậm 2x, 0.25=chậm 4x)")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════
#  VẼ BẢNG THỐNG KÊ LÊN FRAME
# ═══════════════════════════════════════════════════════════════════
def draw_panel(frame, counts, frame_idx, fps, lane_label, split_x, W):
    """
    Vẽ bảng thông tin góc trên trái.

    Kỹ thuật vẽ nền mờ (semi-transparent overlay):
    OpenCV không hỗ trợ vẽ màu trong suốt trực tiếp.
    Trick: copy frame gốc → vẽ hình chữ nhật đặc lên bản copy
           → dùng addWeighted để blend 2 frame lại.
    alpha=0.65 → 65% là bản copy có hộp đen, 35% là frame gốc → nền mờ.
    """
    panel_w = 240
    panel_h = 140 + len(VEHICLE_CLASSES) * 30

    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    cv2.putText(frame, f"LANE: {lane_label.upper()}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)

    elapsed = frame_idx / fps if fps > 0 else 0
    cv2.putText(frame, f"Time: {elapsed:.1f}s", (20, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1, cv2.LINE_AA)

    y = 85
    total = 0
    for cid, name in VEHICLE_CLASSES.items():
        c = counts.get(cid, 0)
        total += c
        cv2.putText(frame, f"{name}:  {c}", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, CLASS_BGR[cid], 1, cv2.LINE_AA)
        y += 28

    cv2.putText(frame, f"TOTAL: {total}", (20, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 180), 2, cv2.LINE_AA)

    # Highlight vùng làn đang đếm bằng lớp màu xanh nhạt (alpha rất thấp = 0.08)
    x1 = 0       if lane_label == "left" else split_x
    x2 = split_x if lane_label == "left" else W
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (x1, 0), (x2, frame.shape[0]), (0, 255, 100), -1)
    cv2.addWeighted(overlay2, 0.08, frame, 0.92, 0, frame)

    return frame


# ═══════════════════════════════════════════════════════════════════
#  VẼ VẠC H ĐẾNG VÀ ĐƯỜNG CHIA LÀN
# ═══════════════════════════════════════════════════════════════════
def draw_scene(frame, line_y, split_x, W, H, lane):
    """
    Vẽ 2 đường tham chiếu lên frame:

    ┌──────────────────────────────────────┐  ← y=0 (trên cùng)
    │  LEFT         │         RIGHT        │
    │               │                      │
    │         ←split_x→                   │
    │               │                      │
    │───────────────┤──────────────────────│  ← line_y (vạch đếm ngang)
    │               │                      │
    └──────────────────────────────────────┘  ← y=H (dưới cùng)

    1. VẠCH ĐẾM (đường NGANG màu đỏ, y = line_y):
       - Xe được đếm khi y-centroid đi qua đường này.
       - Chỉ vẽ trong vùng làn đang được đếm.

    2. ĐƯỜNG CHIA LÀN (đường DỌC màu vàng, x = split_x):
       - Xe có centroid_x < split_x → làn TRÁI.
       - Xe có centroid_x >= split_x → làn PHẢI.
    """
    # Vạch đếm ngang — chỉ trong phần làn đang đếm
    x1 = 0       if lane == "left" else split_x
    x2 = split_x if lane == "left" else W
    cv2.line(frame, (x1, line_y), (x2, line_y), (0, 0, 220), 2)
    cv2.putText(frame, "COUNTING LINE", (x1 + 10, line_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 220), 1, cv2.LINE_AA)

    # Đường chia làn dọc
    cv2.line(frame, (split_x, 0), (split_x, H), (200, 200, 0), 1)
    cv2.putText(frame, "LEFT",  (split_x // 2 - 20, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, "RIGHT", (split_x + (W - split_x) // 2 - 25, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════
#  HÀM CHÍNH
# ═══════════════════════════════════════════════════════════════════
def main():
    args = parse_args()

    # ── 1. Load model YOLO ───────────────────────────────────────
    # ultralytics tự tải file .pt về nếu chưa có (~6MB cho yolov8n).
    # Model chứa weights đã train sẵn trên COCO dataset.
    print(f"[INFO] Loading model: {args.model}")
    model = YOLO(args.model)

    # ── 2. Mở video đầu vào ─────────────────────────────────────
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise FileNotFoundError(f"Không mở được video: {args.input}")

    W            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS          = cap.get(cv2.CAP_PROP_FPS) or 30
    TOTAL_FRAMES = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Video: {W}x{H} @ {FPS:.1f}fps — {TOTAL_FRAMES} frames")

    # ── 3. Chuẩn bị video đầu ra ────────────────────────────────
    fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
    out_fps = max(1.0, FPS * args.speed)
    out     = cv2.VideoWriter(args.output, fourcc, out_fps, (W, H))
    print(f"[INFO] Output speed: {args.speed}x → {out_fps:.1f}fps")

    # ── 4. Tính tọa độ pixel cho các đường ──────────────────────
    #
    #  Dùng tỉ lệ (0.0–1.0) thay vì pixel cứng để code chạy được
    #  trên mọi độ phân giải video mà không cần sửa.
    #
    #  Vạch đếm ngang:
    #    line_y = H * args.line
    #    Ví dụ: H=720, args.line=0.5 → line_y=360 (giữa frame theo chiều cao)
    #
    #  Đường chia làn dọc:
    #    split_x = W * args.split
    #    Ví dụ: W=1280, args.split=0.5 → split_x=640 (giữa frame theo chiều rộng)
    #
    line_y  = int(H * args.line)
    split_x = int(W * args.split)
    print(f"[INFO] Vạch đếm y={line_y}px | Chia làn x={split_x}px | Đếm làn: {args.lane}")

    # ── 5. Khởi tạo ByteTrack ────────────────────────────────────
    # ByteTrack nhận list detection của frame hiện tại, kết nối với
    # detection frame trước bằng Hungarian Algorithm + Kalman Filter,
    # gán track_id bền vững cho từng xe xuyên suốt video.
    byte_tracker    = sv.ByteTrack()
    box_annotator   = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.45, text_thickness=1)

    # ── 6. Khởi tạo biến trạng thái ─────────────────────────────
    counts        = defaultdict(int)   # {class_id: số xe đếm được}
    crossed_ids   = set()              # track_id đã đi qua vạch (tránh đếm 2 lần)
    prev_cy       = {}                 # {track_id: y_centroid frame trước}
    class_history = defaultdict(lambda: deque(maxlen=CLASS_VOTE_WINDOW))

    def voted_class(tid: int) -> int:
        """
        Majority voting: trả về class xuất hiện nhiều nhất trong lịch sử xe tid.

        Ví dụ: class_history[5] = deque([2, 2, 7, 2, 2, 2, 7, 2, 2])
               Counter          = {2: 7, 7: 2}
               most_common(1)   = [(2, 7)]  →  trả về 2 (Car)
        """
        hist = class_history.get(tid)
        if not hist:
            return -1
        return Counter(hist).most_common(1)[0][0]

    def in_target_lane(cx: float) -> bool:
        """Kiểm tra x-centroid có nằm trong làn cần đếm không."""
        if args.lane == "left":
            return cx < split_x
        return cx >= split_x

    # ═══════════════════════════════════════════════════════════════
    #  VÒNG LẶP CHÍNH
    # ═══════════════════════════════════════════════════════════════
    frame_idx = 0
    print("[INFO] Processing...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"  {frame_idx}/{TOTAL_FRAMES} ({frame_idx/TOTAL_FRAMES*100:.1f}%)")

        # ── A. DETECTION ─────────────────────────────────────────
        # YOLO xử lý 1 frame, chỉ tìm class car và truck.
        # iou thấp (0.35) giúp tách các xe đứng cạnh nhau thành box riêng.
        results = model(
            frame,
            classes=VEHICLE_CLASS_IDS,
            conf=args.conf,
            iou=args.iou,
            max_det=100,
            agnostic_nms=False,
            verbose=False,
        )[0]

        detections = sv.Detections.from_ultralytics(results)

        if len(detections) == 0:
            draw_scene(frame, line_y, split_x, W, H, args.lane)
            draw_panel(frame, counts, frame_idx, FPS, args.lane, split_x, W)
            out.write(frame)
            if args.show:
                cv2.imshow("Vehicle Counter — 1 Lane", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            continue

        # ── B. TRACKING ───────────────────────────────────────────
        # ByteTrack thêm tracker_id vào mỗi detection.
        # Cùng 1 xe → cùng 1 tracker_id xuyên suốt video.
        detections = byte_tracker.update_with_detections(detections)

        if detections.tracker_id is None or len(detections) == 0:
            draw_scene(frame, line_y, split_x, W, H, args.lane)
            draw_panel(frame, counts, frame_idx, FPS, args.lane, split_x, W)
            out.write(frame)
            if args.show:
                cv2.imshow("Vehicle Counter — 1 Lane", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            continue

        # ── C. CẬP NHẬT CLASS HISTORY ────────────────────────────
        for i, tid in enumerate(detections.tracker_id):
            class_history[int(tid)].append(int(detections.class_id[i]))

        # ── D. CENTROID CROSSING — Thuật toán đếm xe qua vạch ────
        #
        #  Sơ đồ hoạt động (xe đi từ trên xuống):
        #
        #  Frame N:                 Frame N+1:
        #  ┌──────────────┐         ┌──────────────┐
        #  │  ┌─────┐     │         │              │
        #  │  │  ●──┼─py  │         │  ─────────── │ ← line_y
        #  │  └─────┘     │         │  ┌─────┐     │
        #  │ ─────────────│ ← line_y│  │  ●──┼─cy  │
        #  │              │         │  └─────┘     │
        #  └──────────────┘         └──────────────┘
        #
        #  py < line_y  AND  cy >= line_y  → just_crossed = True
        #
        #  CROSS_BUFFER giúp bắt xe nhanh:
        #  Nếu xe nhảy 10px/frame, không có buffer thì:
        #    Frame N:   py = line_y - 8   (chưa tới line)
        #    Frame N+1: cy = line_y + 8   (đã qua line)
        #  Không có buffer → điều kiện py < line_y → cy >= line_y vẫn đúng.
        #  Nhưng nếu xe nhảy vào điểm CHÍNH XÁC line_y thì buffer giúp chắc hơn.
        #
        boxes = detections.xyxy
        for i, tid in enumerate(detections.tracker_id):
            tid = int(tid)
            x1, y1, x2, y2 = boxes[i]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            if tid in prev_cy:
                py = prev_cy[tid]
                just_crossed = (
                    (py < line_y - CROSS_BUFFER and cy >= line_y - CROSS_BUFFER) or
                    (py > line_y + CROSS_BUFFER and cy <= line_y + CROSS_BUFFER)
                )
                if just_crossed and tid not in crossed_ids and in_target_lane(cx):
                    crossed_ids.add(tid)
                    cid = voted_class(tid)
                    if cid in VEHICLE_CLASSES:
                        counts[cid] += 1

            prev_cy[tid] = cy   # lưu lại để dùng ở frame tiếp theo

        # ── E. ANNOTATE ───────────────────────────────────────────
        labels = []
        for i, tid in enumerate(detections.tracker_id):
            tid  = int(tid)
            x1, y1, x2, y2 = boxes[i]
            cx   = (x1 + x2) / 2.0
            cid  = voted_class(tid)
            name = VEHICLE_CLASSES.get(cid, "Vehicle")
            conf = float(detections.confidence[i])
            in_lane = in_target_lane(cx)
            # ✓ = trong làn đếm | · = làn kia (không đếm)
            labels.append(f"{'✓' if in_lane else '·'} #{tid} {name} {conf:.0%}")

        frame = box_annotator.annotate(frame, detections=detections)
        frame = label_annotator.annotate(frame, detections=detections, labels=labels)

        # Vẽ chấm đỏ ở tâm mỗi bounding box
        for i in range(len(detections)):
            x1, y1, x2, y2 = boxes[i]
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)        # chấm đỏ đặc
            cv2.circle(frame, (cx, cy), 7, (255, 255, 255), 1)     # viền trắng cho dễ nhìn

        draw_scene(frame, line_y, split_x, W, H, args.lane)
        draw_panel(frame, counts, frame_idx, FPS, args.lane, split_x, W)
        out.write(frame)

        if args.show:
            cv2.imshow("Vehicle Counter — 1 Lane", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    out.release()
    if args.show:
        cv2.destroyAllWindows()

    print("\n" + "=" * 40)
    print(f"  KẾT QUẢ — LÀN {args.lane.upper()}")
    print("=" * 40)
    total = 0
    for cid, name in VEHICLE_CLASSES.items():
        c = counts.get(cid, 0)
        total += c
        print(f"  {name:10s}: {c}")
    print(f"  {'TOTAL':10s}: {total}")
    print("=" * 40)
    print(f"[INFO] Output saved: {args.output}")


if __name__ == "__main__":
    main()