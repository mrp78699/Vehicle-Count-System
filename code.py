from ultralytics import YOLO
import cv2
import pandas as pd
import numpy as np
from tracker import *

model = YOLO('yolov8s.pt')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap = cv2.VideoCapture('computer vision/videoplayback.mp4')

my_file = open("C:/Users/Dotnet/Documents/new data/computer vision/Day_4/coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")
print(class_list)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

output_fps = 45  # Set the desired output FPS

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_video.avi', fourcc, output_fps, (frame_width, frame_height))

count = 0
count_1 = 0
lcount = 0
lcount_1 = 0

area_1 = [(480, 280), (240, 280), (240, 287), (480, 287)]
area_2 = [(900, 290), (570, 290), (570, 298), (900, 298)]
tracker = Tracker()
tracker1 = Tracker()
ltracker = Tracker()
ltracker1 = Tracker()
area_c = set()
area_c_1 = set()
larea_c = set()
larea_c_1 = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    frame = cv2.resize(frame, (1020, 500))

    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    list = []
    list_1 = []
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        if 'car' in c:
            list.append([x1, y1, x2, y2])
        elif 'truck' in c:
            list_1.append([x1, y1, x2, y2])
    bbox_id = tracker.update(list)
    abox_id = tracker1.update(list_1)
    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox
        cx = int(x3 + x4) // 2
        cy = int(y3 + y4) // 2
        results = cv2.pointPolygonTest(np.array(area_1, np.int32), ((cx, cy)), False)
        if results >= 0:
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
            cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 2)
            area_c.add(id)

    for abox in abox_id:
        x3, y3, x4, y4, id = abox
        dx = int(x3 + x4) // 2
        dy = int(y3 + y4) // 2
        results = cv2.pointPolygonTest(np.array(area_1, np.int32), ((dx, dy)), False)
        if results >= 0:
            cv2.circle(frame, (dx, dy), 4, (0, 0, 255), -1)
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
            cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 2)
            area_c_1.add(id)

    qx = pd.DataFrame(a).astype("float")
    llist = []
    llist_1 = []
    for index, row in qx.iterrows():
        lx1 = int(row[0])
        ly1 = int(row[1])
        lx2 = int(row[2])
        ly2 = int(row[3])
        ld = int(row[5])
        lc = class_list[ld]
        if 'car' in lc:
            llist.append([lx1, ly1, lx2, ly2])
        elif 'truck' in lc:
            llist_1.append([lx1, ly1, lx2, ly2])
    lbbox_id = ltracker.update(llist)
    labox_id = ltracker1.update(llist_1)
    for lbbox in lbbox_id:
        lx3, ly3, lx4, ly4, lid = lbbox
        lcx = int(lx3 + lx4) // 2
        lcy = int(ly3 + ly4) // 2
        results = cv2.pointPolygonTest(np.array(area_2, np.int32), ((lcx, lcy)), False)
        if results >= 0:
            cv2.circle(frame, (lcx, lcy), 4, (0, 0, 255), -1)
            cv2.rectangle(frame, (lx3, ly3), (lx4, ly4), (0, 0, 255), 2)
            cv2.putText(frame, str(lid), (lx3, ly3), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 2)
            larea_c.add(lid)

    for labox in labox_id:
        lx3, ly3, lx4, ly4, lid = labox
        ldx = int(lx3 + lx4) // 2
        ldy = int(ly3 + ly4) // 2
        results = cv2.pointPolygonTest(np.array(area_2, np.int32), ((ldx, ldy)), False)
        if results >= 0:
            cv2.circle(frame, (ldx, ldy), 4, (0, 0, 255), -1)
            cv2.rectangle(frame, (lx3, ly3), (lx4, ly4), (0, 0, 255), 2)
            cv2.putText(frame, str(lid), (lx3, ly3), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 2)
            larea_c_1.add(lid)

    cv2.polylines(frame, [np.array(area_1, np.int32)], True, (255, 255, 0), 3)
    cv2.polylines(frame, [np.array(area_2, np.int32)], True, (255, 255, 0), 3)
    count = len(area_c)
    count_1 = len(area_c_1)
    lcount = len(larea_c)
    lcount_1 = len(larea_c_1)

    cv2.putText(frame, "LEFT SIDE", (30, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 3)
    cv2.putText(frame, "Car :" + str(count), (30, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 3)
    cv2.putText(frame, "Truck :" + str(count_1), (30, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 3)

    cv2.putText(frame, "RIGHT SIDE", (800, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 3)
    cv2.putText(frame, "Car :" + str(lcount), (800, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 3)
    cv2.putText(frame, "Truck :" + str(lcount_1), (800, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 3)
    
    cv2.imshow("RGB", frame)
    out.write(frame)  # Write the frame to the output video

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()  # Release the VideoWriter object
cv2.destroyAllWindows()
