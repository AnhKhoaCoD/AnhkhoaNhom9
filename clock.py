#Vẽ mặt đồng hồ hình tròn, nền màu tím, có các số dạng la mã màu sắc khác nhau. 
#Có 3 kim đồng hồ: Giờ, phút, giây.
#Kim giờ màu xanh dương, kim phút màu xanh lá cây, kim giây màu đỏ.
#level 2: Vẽ kim giây chuyển động.
#level 3: Vẽ kim phút chuyển động.
#level 4: Vẽ kim giờ chuyển động.   
#level 5: Vẽ thêm các vạch chỉ phút trên mặt đồng hồ. Và kim giây, kim giờ, kim phút, hoạt động theo logic.
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from datetime import datetime

# ===== Khởi tạo =====
clock_base = np.zeros((600, 600, 3), dtype=np.uint8)
clock_base[:] = (0, 0, 25)  #
center = (300, 300)
radius = 250    

# ===== Vẽ mặt đồng hồ =====
cv.circle(clock_base, center, radius, (0, 0, 0), 5)

# ===== Số La Mã =====
roman_numerals = ['XII','I','II','III','IV','V','VI','VII','VIII','IX','X','XI']
for i, numeral in enumerate(roman_numerals):
    angle = i * 30
    x = int(center[0] + (radius - 40) * np.sin(np.radians(angle)))
    y = int(center[1] - (radius - 40) * np.cos(np.radians(angle)))
    cv.putText(clock_base, numeral, (x - 15, y + 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

# ===== Vạch phút =====
def draw_minute_marks(img):
    for i in range(60):
        angle = i * 6
        length = 20 if i % 5 == 0 else 10
        thickness = 3 if i % 5 == 0 else 1

        x1 = int(center[0] + (radius - length) * np.sin(np.radians(angle)))
        y1 = int(center[1] - (radius - length) * np.cos(np.radians(angle)))
        x2 = int(center[0] + radius * np.sin(np.radians(angle)))
        y2 = int(center[1] - radius * np.cos(np.radians(angle)))

        cv.line(img, (x1, y1), (x2, y2), (0,0,0), thickness)

# ===== Vẽ kim =====
def draw_hand(img, angle, length, color, thickness):
    angle -= 90
    x = int(center[0] + length * np.cos(np.radians(angle)))
    y = int(center[1] + length * np.sin(np.radians(angle)))
    cv.line(img, center, (x, y), color, thickness)

# ===== Vòng lặp chính =====
while True:
    clock = clock_base.copy()
    draw_minute_marks(clock)

    now = datetime.now()
    sec = now.second
    minute = now.minute
    hour = now.hour % 12

    sec_angle = sec * 6
    min_angle = minute * 6 + sec * 0.1
    hour_angle = hour * 30 + minute * 0.5

    draw_hand(clock, hour_angle, 120, (255, 0, 0), 8)    # Kim giờ
    draw_hand(clock, min_angle, 170, (0, 255, 0), 6)    # Kim phút
    draw_hand(clock, sec_angle, 200, (0, 0, 255), 2)    # Kim giây

    cv.imshow("Clock", clock)
    if cv.waitKey(1000) & 0xFF == 27:
        break

cv.destroyAllWindows()
