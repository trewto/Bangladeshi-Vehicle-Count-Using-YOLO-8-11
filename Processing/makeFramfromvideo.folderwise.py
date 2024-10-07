import cv2
import os

# Path to the folder containing the video files
#video_folder = 'C:/Users/User/Downloads/YVIDEO/singho/Shahbagh To Banglamotor/training'  # Ensure this is correct
video_folder = 'C:/Users/User/Downloads/YVIDEO/singho/Kakrail to Mogbazar/Training'  # Ensure this is correct

# Path to the folder where you want to save the frames
output_folder = './jotimoy/KakrailtoMogbazar'  # Ensure this is correct

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Define the interval (in seconds) to save frames
interval = 60

# Loop through all files in the video folder
for video_file in os.listdir(video_folder):
    if video_file.endswith('.mp4'):  # Process only .mp4 files
        video_path = os.path.join(video_folder, video_file)
        
        # Open the video file
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Error: Could not open video file {video_file}.")
            continue

        # Get the frame rate (frames per second)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get the total number of frames
        duration = total_frames / fps  # Calculate video duration in seconds
        
        print(f"Processing video: {video_file}, FPS: {fps}, Duration: {duration} seconds")
        
        saved_frame_count = 0
        base_video_name = os.path.splitext(video_file)[0]  # Get the video name without extension
        
        # Save one frame every `interval` seconds, up to the length of the video
        for t in range(0, int(duration), interval):
            # Calculate the corresponding frame for the timestamp
            frame_number = int(t * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()

            if not ret:
                break  # Exit loop if video ends or cannot read frame
            
            # Save the frame with the timestamp in seconds
            frame_filename = os.path.join(output_folder, f'{base_video_name}_{t}.jpg')
            cv2.imwrite(frame_filename, frame)
            print(f"Saved: {frame_filename}")
            saved_frame_count += 1

        # Release the video capture object
        cap.release()
        print(f"Extracted {saved_frame_count} frames from {video_file}")

print("All videos processed.")
