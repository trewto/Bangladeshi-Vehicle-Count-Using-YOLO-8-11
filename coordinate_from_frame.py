import cv2

# Variables to store coordinates
line_start = None
line_end = None

def click_event(event, x, y, flags, params):
    global line_start, line_end
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if line_start is None:
            line_start = (x, y)
            print(f"Line Start set to: {line_start}")
        elif line_end is None:
            line_end = (x, y)
            print(f"Line End set to: {line_end}")


def main():
    video_path = './Processing/Katabon_Intersection_720p.mp4'
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Read the first frame
    success, frame = cap.read()
    if not success:
        print("Error: Could not read the frame.")
        cap.release()
        return

    # Show the frame
    cv2.imshow("Frame", frame)
    print("Click on the video frame to select the line coordinates.")
    print("Left-click to select the start and end points of the line.")
    
    # Set the mouse callback
    cv2.setMouseCallback("Frame", click_event)
    
    # Wait until both points are set
    while line_end is None:
        cv2.waitKey(1)
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    
    if line_start and line_end:
        print(f"LINE_START: {line_start}")
        print(f"LINE_END: {line_end}")
    else:
        print("Error: Could not determine line coordinates.")

if __name__ == "__main__":
    main()
