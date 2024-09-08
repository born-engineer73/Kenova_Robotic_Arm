import cv2
import numpy as np


def adjust_frame_count(frames, desired_length):
    current_length = len(frames)
    if current_length > desired_length:
        # Trim the frames
        frames = frames[:desired_length]
    elif current_length < desired_length:
        # Pad the frames by repeating the last frame
        last_frame = frames[-1]
        padding = [last_frame] * (desired_length - current_length)
        frames.extend(padding)
    return frames

# Function to save video from frames
def save_video(frames, output_path, fps):
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()

# Function to merge videos
def merge_videos(video_paths, output_path, fps):
    merged_frames = []
    
    for video_path in video_paths:
        cap = cv2.VideoCapture(video_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            merged_frames.append(frame)
        cap.release()

    if len(merged_frames) > 0:
        save_video(merged_frames, output_path, fps)

# Specify the path to the video file
video_path = "Data/4reach_3pick_5place_3release_4retract.mp4"

# Initialize video capture from the file
videoCam = cv2.VideoCapture(video_path)

if not videoCam.isOpened():
    print("Error: Cannot open video file")
    exit()

# Read the first frame
ret, frame = videoCam.read()

if not ret:
    print("Error: Cannot read video file")
    exit()

# Select the ROI (Region of Interest) for the bottle
bbox = cv2.selectROI("Tracking", frame, False)

# Initialize CSRT tracker
tracker = cv2.TrackerCSRT_create()
tracker.init(frame, bbox)

# Initialize initial coordinates for reference
initial_x, initial_y = bbox[0], bbox[1]

# Initialize a list to store detected actions with frame numbers
actions = []

# Frame counter
frame_number = 0

# Variables for stability detection
stability_threshold = 10  # Number of consecutive frames to check for stability
stability_counter = 0
stable_x, stable_y = None, None

while True:
    ret, frame = videoCam.read()
    if not ret:
        break

    # Increment frame counter
    frame_number += 1

    # Update tracker
    ret, bbox = tracker.update(frame)

    if ret:
        x, y, w, h = map(int, bbox)
        print(f"Frame: {frame_number}, Object Coordinates: x = {x}, y = {y}")

        # Logic to detect "Pick" action
        if initial_y - y >= 20 and not actions:
            actions.append((frame_number, "Pick"))
            initial_y = y  # Update initial_y to current y for future reference
            print(f"Frame: {frame_number}, Action Detected: Pick")

        # Logic to detect "Move" action
        if initial_x - x >= 20 and actions and actions[-1][1] == "Pick":
            actions.append((frame_number, "Move"))
            initial_x = x  # Update initial_x to current x for future reference
            initial_y = y 
            print(f"Frame: {frame_number}, Action Detected: Move")

        if  y- initial_y >= 17 and actions and actions[-1][1] == "Move":
            actions.append((frame_number, "Place"))
            initial_y = y  # Update initial_y to current y for future reference
            print(f"Frame: {frame_number}, Action Detected: Place")

        # Logic to detect "Place" action
        if actions and actions[-1][1] == "Place":
            # Initialize stability variables
            if stable_x is None or stable_y is None:
                stable_x, stable_y = x, y
                stability_counter = 0

            # Check if coordinates are stable
            if abs(x - stable_x) < 5 and abs(y - stable_y) < 5:
                stability_counter += 1
                if stability_counter >= stability_threshold:
                    actions.append((frame_number, "Retract"))
                    stable_x, stable_y = None, None  # Reset stability variables
                    print(f"Frame: {frame_number}, Action Detected: Retract")
            else:
                # Reset stability counter if coordinates are not stable
                stability_counter = 0
                stable_x, stable_y = x, y

        # Draw the rectangle around the tracked object
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Display the coordinates on the video frame
        coord_text = f"({x}, {y})"
        cv2.putText(frame, coord_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
    else:
        cv2.putText(frame, "Tracking failed", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("Tracking", frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Add start and end frames to `actions` list
total_frames = int(videoCam.get(cv2.CAP_PROP_FRAME_COUNT))
actions.insert(0, (0, "Start"))  # Add start frame
actions.append((total_frames - 1, "End"))  # Add end frame

# Define the desired frame lengths (90, 120, 150)
desired_frame_lengths = [90, 120, 150]

# Create a list to store frame ranges for each action
action_segments = []

# Calculate the video segments
for i in range(1, len(actions)):
    start_frame = actions[i-1][0]
    end_frame = actions[i][0] - 10
    action_segments.append((start_frame, end_frame, actions[i-1][1]))

# Open the video again to read frames
videoCam.set(cv2.CAP_PROP_POS_FRAMES, 0)
fps = videoCam.get(cv2.CAP_PROP_FPS)

# List to store paths of saved videos for merging later
saved_videos = []

# Extract and save video segments
for segment in action_segments:
    start_frame, end_frame, action_name = segment
    frames = []
    videoCam.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    for _ in range(start_frame, end_frame + 1):
        ret, frame = videoCam.read()
        if not ret:
            break
        frames.append(frame)

    # Determine the closest desired frame length
    segment_length = len(frames)
    closest_length = min(desired_frame_lengths, key=lambda x: abs(x - segment_length))

    # Adjust the frames to the desired length
    adjusted_frames = adjust_frame_count(frames, closest_length)

    # Save the segment as a video
    output_path = f"./result/{action_name.lower()}.mp4"
    save_video(adjusted_frames, output_path, fps)
    saved_videos.append(output_path)
    print(f"Saved {action_name} video with {closest_length} frames to {output_path}")

# Merge all the saved videos into one final result
final_output_path = "./result/final_result.mp4"
merge_videos(saved_videos, final_output_path, fps)
print(f"Merged video saved as {final_output_path}")

videoCam.release()
cv2.destroyAllWindows()

# Print the detected actions with frame numbers
print("Detected Actions:", actions)
