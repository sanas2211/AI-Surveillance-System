import cv2

# define zone (rectangle)
ZONE = (100, 100, 400, 400)

def draw_zone(frame):
    x1, y1, x2, y2 = ZONE
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)
    return frame

def check_intrusion(box):
    x1, y1, x2, y2 = ZONE
    bx1, by1, bx2, by2 = box

    if bx1 > x1 and by1 > y1 and bx2 < x2 and by2 < y2:
        return True
    return False