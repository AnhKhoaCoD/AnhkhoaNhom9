import cv2 as cv
import numpy as np
import math
import time

# ======================
# THIẾT LẬP
# ======================
size = 800
center = (size // 2, size // 2)
radius = 300
window_name = "Dong ho La Ma - Dep"

roman_nums = [
     "I", "II", "III", "IV", "V", "VI",
     "VII", "VIII", "IX", "X", "XI", "XII",
]

colors = [
    (255, 120, 120), (120, 255, 120), (120, 120, 255),
    (255, 255, 120), (255, 120, 255), (120, 255, 255),
    (230, 230, 230), (180, 255, 180), (255, 180, 180),
    (180, 180, 255), (220, 180, 255), (255, 220, 180)
]

cv.namedWindow(window_name)

# ======================
# VÒNG LẶP CHÍNH
# ======================
while True:
    # ----------------------
    # NỀN TÍM GRADIENT
    # ----------------------
    img = np.zeros((size, size, 3), dtype=np.uint8)
    for i in range(size):
        img[i, :] = (160 + i // 20, 40, 160 + i // 20)

    # ----------------------
    # MẶT ĐỒNG HỒ
    # ----------------------
    cv.circle(img, center, radius, (240, 240, 240), 4)
    cv.circle(img, center, radius - 5, (200, 200, 200), 1)

    # ----------------------
    # LEVEL 5: VẠCH PHÚT / GIỜ
    # ----------------------
    for i in range(60):
        angle = math.radians(i * 6 - 90)
        r1 = radius - 8
        r2 = radius - (28 if i % 5 == 0 else 18)
        thickness = 3 if i % 5 == 0 else 1

        x1 = int(center[0] + math.cos(angle) * r1)
        y1 = int(center[1] + math.sin(angle) * r1)
        x2 = int(center[0] + math.cos(angle) * r2)
        y2 = int(center[1] + math.sin(angle) * r2)

        cv.line(img, (x1, y1), (x2, y2), (255, 255, 255), thickness)

    # ----------------------
    # SỐ LA MÃ (CÂN GIỮA)
    # ----------------------
    for i, num in enumerate(roman_nums):
        angle = math.radians(i * 30 - 60)
        x = int(center[0] + math.cos(angle) * (radius - 55))
        y = int(center[1] + math.sin(angle) * (radius - 55))

        (w, h), _ = cv.getTextSize(num, cv.FONT_HERSHEY_DUPLEX, 0.9, 2)
        cv.putText(
            img, num,
            (x - w // 2, y + h // 2),
            cv.FONT_HERSHEY_DUPLEX,
            0.9,
            colors[i],
            2,
            cv.LINE_AA
        )

    # ----------------------
    # THỜI GIAN THỰC
    # ----------------------
    t = time.localtime()
    hour = t.tm_hour % 12
    minute = t.tm_min
    second = t.tm_sec

    sec_angle = math.radians(second * 6 - 90)
    min_angle = math.radians((minute + second / 60) * 6 - 90)
    hour_angle = math.radians((hour + minute / 60) * 30 - 90)

    # ----------------------
    # KIM ĐỒNG HỒ (ĐẸP HƠN)
    # ----------------------
    # Kim giờ – xanh dương
    cv.line(img, center,
            (int(center[0] + math.cos(hour_angle) * 140),
             int(center[1] + math.sin(hour_angle) * 140)),
            (255, 0, 0), 10)

    # Kim phút – xanh lá
    cv.line(img, center,
            (int(center[0] + math.cos(min_angle) * 210),
             int(center[1] + math.sin(min_angle) * 210)),
            (0, 255, 0), 6)

    # Kim giây – đỏ
    cv.line(img, center,
            (int(center[0] + math.cos(sec_angle) * 260),
             int(center[1] + math.sin(sec_angle) * 260)),
            (0, 0, 255), 2)

    # Tâm đồng hồ
    cv.circle(img, center, 10, (255, 255, 255), -1)
    cv.circle(img, center, 5, (120, 120, 120), -1)

    # ----------------------
    # HIỂN THỊ & THOÁT
    # ----------------------
    cv.imshow(window_name, img)

    if cv.getWindowProperty(window_name, cv.WND_PROP_VISIBLE) < 1:
        break

    key = cv.waitKey(1000)
    if key == 27 or key == ord('q'):
        break

cv.destroyAllWindows()
