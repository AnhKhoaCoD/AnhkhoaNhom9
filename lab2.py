import cv2 as cv
import numpy as np
import math

vid = cv.VideoCapture("bang_chuyen.mp4")

LINE_1 = 600
DIST_THRESHOLD = 25 

tracked_objects = {}
counted_ids = set()
object_id = 0
count = 0


def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


while True:
    ret, frame = vid.read()
    if not ret:
        break

    cv.line(frame, (LINE_1, 0), (LINE_1, frame.shape[0]), (0, 0, 255), 2)

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (5, 5), 0)
    clahed = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahed.apply(gray)

    circles = cv.HoughCircles(
        gray, cv.HOUGH_GRADIENT, dp=1, minDist=20,
        param1=50, param2=18, minRadius=5, maxRadius=50
    )

    current_centers = []

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for x, y, r in circles[0]:
            current_centers.append((x, y))
            cv.circle(frame, (x, y), r, (0, 255, 0), 2)
            cv.circle(frame, (x, y), 2, (0, 0, 255), 3)

    new_tracked = {}
    used_old_ids = set()

    for center in current_centers:
        best_match_id = None
        min_dist = float("inf")

        for oid, prev_center in tracked_objects.items():
            if oid in used_old_ids:
                continue

            d = distance(center, prev_center)
            if d < min_dist and d < DIST_THRESHOLD:
                min_dist = d
                best_match_id = oid

        if best_match_id is not None:
            prev_center = tracked_objects[best_match_id]
            new_tracked[best_match_id] = center
            used_old_ids.add(best_match_id)

            # kiểm tra cắt line
            if best_match_id not in counted_ids and prev_center[0] < LINE_1 <= center[0]:
                count += 1
                counted_ids.add(best_match_id)
                print(f"[Số hình tròn đã đi qua line đỏ] Số lượng = {count}")
        else:
            new_tracked[object_id] = center
            object_id += 1

    tracked_objects = new_tracked


    cv.imshow("Circle Counter", frame)

    if cv.waitKey(10) & 0xFF == ord("q"):
        break

vid.release()
cv.destroyAllWindows()