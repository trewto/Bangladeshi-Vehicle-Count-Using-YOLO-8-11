#this is a stable version
from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
import torch
import time  # Import the time module

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


LINE_START= (796, 111)
LINE_END= (995, 156)


LINE_START= (716, 171)
LINE_END= (1016, 200)

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

banglamotor new
LINE_START: (796, 111)
LINE_END: (995, 156)

"""


#model1 = YOLO("I:/Git/Code-With-Nayeem/Train_With_GPU/runs/detect/train9/weights/best.pt")#### IT IS THE MODEL WITH BANGLAMOTOR 
model1 = YOLO("I:/Git/Code-With-Nayeem/Train_With_GPU/runs/detect/train14/weights/best.pt")
model2 = YOLO("yolov8l.pt")#pretrained
#SOURCE_VIDEO_PATH = './Processing/Katabon_Intersection_720p.mp4'
#SOURCE_VIDEO_PATH = './Processing/Shahbagh_Intersection.mp4'

SOURCE_VIDEO_PATH = './Processing/Banglamotor_Intersection.mp4'




is_show_live = True
write_output = True




both_out_off = False
if is_show_live== False and write_output == False:
    both_out_off = True

    
#########################
#for model 1 
print ("For model 1 ")
model1.to(device)
class_names = model1.names
# Display the number of classes and their names
num_classes = len(class_names)
print(f"Number of classes: {num_classes}")
print("Class names:", class_names)

#########################
#for model 2
print ("For model 2 ")
model2.to(device)
class_names2 = model2.names
num_classes2 = len(class_names)
print(f"Number of classes: {num_classes2}")
print("Class names:", class_names2)

#exit();

if LINE_START and LINE_END:
    print(f"LINE_START: {LINE_START}")
    print(f"LINE_END: {LINE_END}")
else:
    print("Error: Could not determine line coordinates.")
    exit()
def is_overlapping(box1, box2, iou_threshold=0.4):
    """
    Determines whether two bounding boxes overlap by calculating their Intersection Over Union (IoU)
    and checking if one is contained within the other based on area.

    Args:
        box1, box2: Bounding boxes in the format (x_center, y_center, width, height).
        iou_threshold: Minimum IoU to consider the boxes as overlapping. Defaults to 0.5.

    Returns:
        True if the boxes overlap, False otherwise.
    """
    # Extract box1 coordinates
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Calculate the (x, y)-coordinates of the top-left and bottom-right corners for both boxes
    x1_min, y1_min = x1 - w1 / 2, y1 - h1 / 2
    x1_max, y1_max = x1 + w1 / 2, y1 + h1 / 2

    x2_min, y2_min = x2 - w2 / 2, y2 - h2 / 2
    x2_max, y2_max = x2 + w2 / 2, y2 + h2 / 2

    # Calculate areas
    box1_area = w1 * h1
    box2_area = w2 * h2

    # Check if box2 is completely inside box1
    if x1_min <= x2_min <= x2_max <= x1_max and y1_min <= y2_min <= y2_max <= y1_max:
        # Calculate the ratio of the inner box area to the outer box area
       
        return (box2_area / box1_area)>= iou_threshold  # box2 is inside box1 and meets the area ratio requirement
    

    # Check if box1 is completely inside box2
    if x2_min <= x1_min <= x1_max <= x2_max and y2_min <= y1_min <= y1_max <= y2_max:
        # Calculate the ratio of the inner box area to the outer box area
       
       return (box1_area / box2_area) >= iou_threshold  # box1 is inside box2 and meets the area ratio requirement
    

    # Calculate the (x, y)-coordinates of the intersection rectangle
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    # Compute the width and height of the intersection rectangle
    inter_width = max(0, inter_x_max - inter_x_min)
    inter_height = max(0, inter_y_max - inter_y_min)

    # Compute the area of intersection
    inter_area = inter_width * inter_height

    # Compute the union area (total area covered by both boxes)
    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area

    # Check if the IoU exceeds the threshold
    return iou >= iou_threshold



cap = cv2.VideoCapture(SOURCE_VIDEO_PATH)

track_history1 = defaultdict(lambda: [])
track_history2 = defaultdict(lambda: [])

line_crossing_counts1 = defaultdict(int)  # To keep track of line crossing
line_crossing_counts2 = defaultdict(int)  # To keep track of line crossing


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


line_crossing_counts1 = defaultdict(lambda: defaultdict(int))  # To keep track of line crossing counts by class and track_id
line_crossing_counts2 = defaultdict(lambda: defaultdict(int))  # To keep track of line crossing counts by class and track_id


total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


print(f"Total number of frames in the video: {total_frames}")
start_time= time.time()


#point_recorded = [];
with sv.VideoSink("output_single_line.mp4", video_info) as sink:
    
    while cap.isOpened():
        
        success, frame = cap.read()
        frame_count += 1
        
     
        #if frame_count%2 == 0: 
        #    continue
        
        
        if not success:
            print("Error: Could not read frame.")
            break

        if success:
            
            results1 = model1.track(frame, persist=True, verbose=False, classes=[0,1])
            results2 = model2.track(frame, persist=True, verbose=False, classes=[2,3,5])
           

            # Processing for model 1 
            boxes1 = results1[0].boxes.xywh.to(device)
            track_ids1 = results1[0].boxes.id
            class_ids1 = results1[0].boxes.cls.int().tolist()
            confidences1 = results1[0].boxes.conf  # Fix: use `boxes.conf` instead of `results1[0].scores`
            
            
            # Processing for model 2 
            boxes2 = results2[0].boxes.xywh.to(device)
            track_ids2 = results2[0].boxes.id
            class_ids2 = results2[0].boxes.cls.int().tolist()
            confidences2 = results2[0].boxes.conf  # Fix: use `boxes.conf` instead of `results1[0].scores`

            # Create a copy of the frame to draw the annotations
            annotated_frame = frame.copy()

            if track_ids1 is not None: 
                # Draw bounding boxes for model 1
                for box1,track_id1, class_id1,cf in zip(boxes1, track_ids1,class_ids1,confidences1):
                    x, y, w, h = box1 
                    track_id1 = int(track_id1)
                    class_id1 = int(class_id1)
                    top_left = (int(x - w / 2), int(y - h / 2))
                    bottom_right = (int(x + w / 2), int(y + h / 2))
                    conf = f"{cf:.2f}"
                    track1 = track_history1[track_id1]
                    track1.append((float(x), float(y)))
                    if len(track1) > 20:
                        track1.pop(0)
                     # Draw the movement track

                    if both_out_off is False:
                        points1 = np.array(track1).astype(np.int32).reshape((-1, 1, 2))
                        cv2.polylines(annotated_frame, [points1], isClosed=False, color=(230, 230, 230), thickness=2)


                    color = (0, 255, 0)  # Green for model 1
                    if both_out_off is False:
                        cv2.rectangle(annotated_frame, top_left, bottom_right, color, 2)
                        cv2.putText(annotated_frame, f"{model1.names[class_id1]}_{class_id1}, T-{track_id1}, {conf}", (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    if len(track1) > 1:
                        prev_x = track1[-2][0]
                        prev_y = track1[-2][1] 
                        if two_line_intersect(x, y, prev_x, prev_y, LINE_START[0], LINE_START[1], LINE_END[0], LINE_END[1]):
                            line_crossing_counts1[class_id1][track_id1] = 1
                            
            if both_out_off is False:
                output_string = " "
                for class_id1, counts1 in line_crossing_counts1.items():
                    class_name1 = model1.names[class_id1]
                    output_string += f"{class_id1}_{class_name1}={sum(counts1.values())},"


                # Draw the counting line
            if both_out_off is False:
                total_count = sum([sum(counts.values()) for counts in line_crossing_counts1.values()])
                    #cv2.putText(annotated_frame, f"Total: {total_count} , Frame = {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(annotated_frame, f"Total: {total_count} ,  Frame = {frame_count} ::: {output_string}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    

            if track_ids2 is not None: 
                # Draw bounding boxes for model 2
                for box2,track_id2, class_id2,cf in zip(boxes2,track_ids2, class_ids2,confidences2):

                   # if is_overlapping(box1, box2):
                    #    continue # Skip drawing the bounding box if it overlaps with a bounding box from model 1
                    #if is_overlapping(box2, box1):
                    #    continue
                    #if is_overlapping(box2, box1):
                    #    continue # Skip drawing the bounding box if it overlaps with a bounding box from model 1
                    con = 0 ; 
                    if boxes1 is not None:
                        for box1 in boxes1:
                            if is_overlapping(box1, box2):
                                con = 1 

                    if con == 1:
                        continue 
                        


                    x, y, w, h = box2
                    track_id2 = int(track_id2)
                    class_id2 = int(class_id2)
                    cf = f"{cf:.2f}"
                    top_left = (int(x - w / 2), int(y - h / 2))
                    bottom_right = (int(x + w / 2), int(y + h / 2))

                    track2 = track_history2[track_id2]
                    track2.append((float(x), float(y)))
                    if len(track2) > 20:
                        track2.pop(0)
                     # Draw the movement track
                    if both_out_off is False:
                        points2 = np.array(track2).astype(np.int32).reshape((-1, 1, 2))
                        cv2.polylines(annotated_frame, [points2], isClosed=False, color=(230, 230, 230), thickness=2)




                  
                    color = (0, 0, 255)  # Red for model 2
                    if both_out_off is False:
                        cv2.rectangle(annotated_frame, top_left, bottom_right, color, 2)
                        cv2.putText(annotated_frame, f"{model2.names[class_id2]}_{class_id2}, T-{track_id2}, {cf}", (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    if len(track2) > 1:
                        prev_x = track2[-2][0]
                        prev_y = track2[-2][1] 
                        if two_line_intersect(x, y, prev_x, prev_y, LINE_START[0], LINE_START[1], LINE_END[0], LINE_END[1]):
                            line_crossing_counts2[class_id2][track_id2] = 1
                    

          
                                        
            
            if both_out_off is False:
                output_string = " "
                for class_id2, counts2 in line_crossing_counts2.items():
                    class_name2 = model2.names[class_id2]
                    output_string += f"{class_id2}_{class_name2}={sum(counts2.values())},"


                    # Draw the counting line
                total_count = sum([sum(counts.values()) for counts in line_crossing_counts2.values()])
                cv2.putText(annotated_frame, f"Total: {total_count} ,  Frame = {frame_count} ::: {output_string}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2) 
                    

        else:
            break
        
        if both_out_off is False:
            cv2.line(annotated_frame, LINE_START, LINE_END, (0, 0, 255), 2)

        if is_show_live:
            cv2.imshow("YOLOv11 Tracking", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        
        if write_output:
            sink.write_frame(annotated_frame)
       
        
       
if write_output:
    cap.release()
    cv2.destroyAllWindows()
#print(f"Total objects crossed the line: {sum(line_crossing_counts.values())}")

end_time = time.time()
print ("Time taken: ", end_time-start_time)
print("Processing complete.")
print (model1)
print (model2)
print (SOURCE_VIDEO_PATH)
for class_id, counts in line_crossing_counts1.items():
    class_name = model1.names[class_id]
    print(f"Total objects of class {class_id} {class_name} crossed the line: {sum(counts.values())}")
for class_id, counts in line_crossing_counts2.items():
    class_name = model2.names[class_id]
    print(f"Total objects of class {class_id} {class_name} crossed the line: {sum(counts.values())}")


