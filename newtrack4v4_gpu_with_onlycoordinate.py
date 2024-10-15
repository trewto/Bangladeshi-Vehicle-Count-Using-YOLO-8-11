#this is a stable version
from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#LINE_START: (9, 509)
#LINE_END: (1050, 350)

LINE_START = (9, 509) 
LINE_END = (1050, 350)

#LINE_START= (68, 515)
#LINE_END= (798, 535)
LINE_START= (758, 179)
LINE_END= (972, 209)
LINE_START= (9, 509)
LINE_END= (1050, 350)

"""
katabon
LINE_START: (9, 509)
LINE_END: (1050, 350)


LINE_START: (68, 515)
LINE_END: (798, 535)
Location: Shahbag, shahbag23y8


LINE_START: (716, 171)
LINE_END: (1016, 200)
Location: Banglamotor, banglamotor

"""


#model = YOLO("I:/Git/Code-With-Nayeem/Train_With_GPU/runs/detect/train9/weights/best.pt")
#model = YOLO("I:/Git/Code-With-Nayeem/Train_With_GPU__v2_Same_Weight/runs/detect/train2/weights/best.pt")
model = YOLO("I:/Git/Code-With-Nayeem/Train_With_GPU__v2_Same_Weight/runs/detect/train2/weights/best.pt")
#model = YOLO("I:/Git/Code-With-Nayeem/Train_With_GPU__v2_Same_Weight/runs/detect/train2/weights/best.pt").load("yolo11n.pt")
#model = YOLO("yolo11s.pt")
#model = YOLO("train6best___.pt").to(device).half()

SOURCE_VIDEO_PATH = './Processing/Katabon_Intersection_720p.mp4'
#SOURCE_VIDEO_PATH = './Processing/Shahbagh_Intersection.mp4'
#SOURCE_VIDEO_PATH = './Processing/Banglamotor_Intersection.mp4'
#SOURCE_VIDEO_PATH = './Processing/4K Video of Highway Traffic.mp4'
#SOURCE_VIDEO_PATH = './Processing/obdetcfromyoutbue.mp4'
#SOURCE_VIDEO_PATH = './Processing/061.mp4'
#SOURCE_VIDEO_PATH = './Processing/oneway.mp4'



model.to(device)
class_names = model.names

# Display the number of classes and their names
num_classes = len(class_names)
print(f"Number of classes: {num_classes}")
print("Class names:", class_names)






if LINE_START and LINE_END:
    print(f"LINE_START: {LINE_START}")
    print(f"LINE_END: {LINE_END}")
else:
    print("Error: Could not determine line coordinates.")
    exit()





cap = cv2.VideoCapture(SOURCE_VIDEO_PATH)

track_history = defaultdict(lambda: [])

line_crossing_counts = defaultdict(int)  # To keep track of line crossing
def two_line_intersect(x1, y1, x2, y2, x3, y3, x4, y4):
    # Check if the two lines are vertical
    if x1 == x2 and x3 == x4:
        # Both lines are vertical and parallel (no intersection unless they are the same line)
        return False
    elif x1 == x2:
        # Line 1 is vertical
        x_intersect = x1
        m2 = (y4 - y3) / (x4 - x3)
        c2 = y3 - m2 * x3
        y_intersect = m2 * x_intersect + c2
    elif x3 == x4:
        # Line 2 is vertical
        x_intersect = x3
        m1 = (y2 - y1) / (x2 - x1)
        c1 = y1 - m1 * x1
        y_intersect = m1 * x_intersect + c1
    else:
        # Calculate the slopes of the lines
        m1 = (y2 - y1) / (x2 - x1)
        m2 = (y4 - y3) / (x4 - x3)
        # Calculate the y-intercepts of the lines
        c1 = y1 - m1 * x1
        c2 = y3 - m2 * x3
        # Check if the lines are parallel
        if m1 == m2:
            return False
        # Calculate the intersection point
        x_intersect = (c2 - c1) / (m1 - m2)
        y_intersect = m1 * x_intersect + c1

    # Check if the intersection point lies on both lines
    if (min(x1, x2) <= x_intersect <= max(x1, x2)) and (min(x3, x4) <= x_intersect <= max(x3, x4)) and \
       (min(y1, y2) <= y_intersect <= max(y1, y2)) and (min(y3, y4) <= y_intersect <= max(y3, y4)):
        return True
    return False



frame_count = 0
video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
print("Starting processing...")
line_crossing_counts = defaultdict(lambda: defaultdict(int))  # To keep track of line crossing counts by class and track_id
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Total number of frames in the video: {total_frames}")
point_recorded = [];
with sv.VideoSink("output_single_line.mp4", video_info) as sink:

    while cap.isOpened():
        success, frame = cap.read()
        frame_count += 1
        #if  i pres q it will break
        #optimization included 
        #if frame_count%2 == 0 :
        #    continue

        #if frame_count%1000 ==0 : 
        #    print(frame_count)
            
        
        #if frame_count%20000 == 0 or frame_count%200001 == 0:
        #    print(f"Processing frame {frame_count}, Continue? ")
        #if q is press i 5 second it will break , otherwise continue 
        #    if cv2.waitKey(1) & 0xFF == ord("q"):
        ##       break
       #     i = input("Press Enter to continue...")
        #    if i == 'q':
        #        break
        if frame_count%500 == 0:
            print(frame_count)
        if frame_count%2 == 0: 
            continue
        

        
        
        if not success:
            print("Error: Could not read frame.")
            break

        if success:
            #results = model.track(frame, persist=True, verbose=False, classes=[0,1,2,3,4])
            #results = model.track(frame, persist=True, verbose=False,conf=0.2)
            #results = model.track(frame, persist=True, verbose=False,conf=0.2,classes=[2,3,5])
           # results = model.track(frame, persist=True, verbose=False,classes=[2,3,5])
           # results = model.track(frame, persist=True, verbose=False,classes=[2,3,5,80,81])
            results = model.track(frame, persist=True, verbose=False)
           # results = model.track(frame, persist=True, verbose=False)
            #boxes = results[0].boxes.xywh.cpu()
            boxes = results[0].boxes.xywh.to(device)
            track_ids = results[0].boxes.id
            class_ids = results[0].boxes.cls.int().tolist()
            annotated_frame = results[0].plot()
            if track_ids is not None:
                track_ids = track_ids.int().tolist()
                #annotated_frame = results[0].plot()

                for box, track_id,class_id in zip(boxes, track_ids,class_ids):
                    class_id = int(class_id)
                    x, y, w, h = box
                    track = track_history[track_id]
                    track.append((float(x), float(y)))
                    #point_recorded.append((float(x), float(y)))


                    if len(track) > 100:
                        track.pop(0)
                
                    # Draw the movement track
                    points = np.array(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=2)

                    # Check if the object has crossed the line
                    if len(track) > 1:
                        #print(f"Average y value for track {track_id}: {average_y}")
                        
                        # Check if the object has crossed the line
                        #if  abs(x - LINE_START[0]) < 30:
                        #    line_crossing_counts[track_id] = 1
                        #    print(f"Object {track_id} crossed the line. Count: {line_crossing_counts[track_id]}")
                        prev_x = track[-2][0]
                        #average_x = (prev_x + x) / 2
                        #print(f"Average x value for track {track_id}: {average_x}")
                        prev_y = track[-2][1] 
                        #average_y = (prev_y + y) / 2

                        #now  i want to draw a line (x,y) to (prev_x, prev_y) and check if it crosses the line ( LINE_START[0], LINE_START[0]) to ( LINE_END[0], LINE_END[0])

                        if two_line_intersect(x, y, prev_x, prev_y, LINE_START[0], LINE_START[1], LINE_END[0], LINE_END[1]):
                            line_crossing_counts[class_id][track_id] = 1
                            #print(f"Object {track_id} crossed the line. Count: {line_crossing_counts[track_id]}")

                        # Check if the object has crossed the line
                        #if y < LINE_START[1] and prev_y >= LINE_START[1]:
                        #    line_crossing_counts[track_id] += 1
                        #    print(f"Object {track_id} crossed the line. Count: {line_crossing_counts[track_id]}")
                    

                    # Draw the counting line
                    total_count = sum([sum(counts.values()) for counts in line_crossing_counts.values()])
                    cv2.putText(annotated_frame, f"Total: {total_count} , Frame = {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    

                #cv2.imshow("YOLOv8 Tracking", annotated_frame)
                #if cv2.waitKey(1) & 0xFF == ord("q"):
                #    break
            else:
                #print("No objects detected or tracked in this frame.")
                x = 1; #do nothing
                #q
                # print("x =1")

        else:
            break
        # Write the frame with annotations to the output video
        #points = np.array(point_recorded).astype(np.int32).reshape((-1, 1, 2))
        #cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=2)
        # Plot the points
        #for (x, y) in point_recorded:
        #    cv2.circle(annotated_frame, (int(x), int(y)), radius=2, color=(0, 0, 255), thickness=-1)

        cv2.line(annotated_frame, LINE_START, LINE_END, (0, 0, 255), 2)
        cv2.imshow("YOLOv8 Tracking", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        sink.write_frame(annotated_frame)

cap.release()
cv2.destroyAllWindows()
#print(f"Total objects crossed the line: {sum(line_crossing_counts.values())}")
print("Processing complete.")


total_count = sum([sum(counts.values()) for counts in line_crossing_counts.values()])
print(f"Total objects crossed the line: {total_count}")
for class_id, counts in line_crossing_counts.items():
    class_name = model.names[class_id]
    print(f"Total objects of class {class_id} {class_name} crossed the line: {sum(counts.values())}")

