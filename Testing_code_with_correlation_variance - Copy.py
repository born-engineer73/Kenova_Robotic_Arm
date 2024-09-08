#######################################################
''' code for significant correlation change between the consequtive frames but isuue is it will detect 149 to 150 then 150 to 151 that we don't want'''

# import os
# import cv2
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# import time

# # Load the trained model
# try:
#     model = load_model('Saved_models/New_subtasks_VGG16(26_3-06-2024).h5')
# except Exception as e:
#     print(f"Error loading model: {e}")
#     exit(1)

# def min_max_normalize(data): 
#     min_val = np.min(data)
#     max_val = np.max(data)
#     normalized_data = (data - min_val) / (max_val - min_val)
#     return normalized_data.reshape(224, 224, 3)

# # Function to extract frames from a video and calculate variance image
# def calculate_pixel_variance(frames):
#     num_frames = len(frames)
#     frame_height, frame_width = frames[0].shape[:2]

#     # Initialize an array to hold the variances
#     variance_array = np.zeros((frame_height, frame_width, 3))

#     # Iterate over each pixel position
#     for i in range(frame_height):
#         for j in range(frame_width):
#             # Extract the pixel values across all frames for the current pixel position
#             pixel_values = [frames[k][i, j] for k in range(num_frames)]

#             # Calculate the variance of the pixel values
#             variance_array[i, j] = np.var(pixel_values, axis=0)

#     return variance_array

# # Function to calculate the correlation between two frames
# def calculate_frame_correlation(frame1, frame2):
#     frame1_flat = frame1.flatten()
#     frame2_flat = frame2.flatten()
#     correlation = np.corrcoef(frame1_flat, frame2_flat)[0, 1]
#     return correlation

# # Function to compute and print correlation between consecutive frames in a video

# def print_frame_correlations(video_path, target_size=(224, 224), threshold=0.2):
#     cap = cv2.VideoCapture(video_path)
#     ret, prev_frame = cap.read()
#     if not ret:
#         print("Error reading video")
#         cap.release()
#         return
    
#     prev_frame = cv2.resize(prev_frame, target_size)
#     prev_frame = prev_frame / 255.0

#     frame_idx = 1
#     prev_correlation = None
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         frame = cv2.resize(frame, target_size)
#         frame = frame / 255.0
        
#         correlation = calculate_frame_correlation(prev_frame, frame)
#         print(f"Correlation between frame {frame_idx-1} and frame {frame_idx}: {correlation}")
        
#         if prev_correlation is not None:
#             abs_diff = abs(correlation - prev_correlation)
#             if abs_diff > threshold:
#                 print(f"Significant change in correlation at frame {frame_idx}: {abs_diff}")
        
#         prev_frame = frame
#         prev_correlation = correlation
#         frame_idx += 1

#     cap.release()

# # Function to predict labels for a given video
# def predict_video_labels(video_path, segment_size=150, target_size=(224, 224)):
#     cap = cv2.VideoCapture(video_path)
#     frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     print(frame_count)
#     total_frames = frame_count
    
#     labels = []

#     for start_frame in range(0, total_frames, segment_size):
#         video_frames = []
#         for i in range(start_frame, start_frame + segment_size):
#             if i >= total_frames: 
#                 break
#             cap.set(cv2.CAP_PROP_POS_FRAMES, i)
#             ret, frame = cap.read()
#             if not ret:
#                 continue
#             frame = cv2.resize(frame, target_size)
#             frame = frame / 255.0
#             video_frames.append(frame)
        
#         if len(video_frames) <= segment_size:
#             variance_frame = calculate_pixel_variance(video_frames)
#             variance_frame = min_max_normalize(variance_frame)
#             variance_frame = np.expand_dims(variance_frame, axis=0)  # Normalize and add batch dimension
#             predicted_label = np.argmax(model.predict(variance_frame), axis=-1)
#             labels.append(predicted_label[0])
#     cap.release()
#     return labels

# def capture_and_save_video(video_path):
#     cap = cv2.VideoCapture(0)  # Use the default webcam (change to the appropriate index if you have multiple webcams)
#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

#     start_time = time.time()

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         elapsed_time = time.time() - start_time
#         minutes = int(elapsed_time // 60)
#         seconds = int(elapsed_time % 60)
#         timer_text = f'Time: {minutes:02}:{seconds:02}'

#         # Add the timer text to the frame
#         cv2.putText(frame, timer_text, (10, frame_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

#         out.write(frame)

#         # Display the captured frame
#         cv2.imshow('frame', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to stop capturing
#             break

#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()

# # Example usage
# video_path = 'Video_for_testing/27_6_merge_5.mp4'

# # To capture video and save it, you can call capture_and_save_video
# # capture_and_save_video(video_path)

# # Compute and print correlations between consecutive frames
# print_frame_correlations(video_path, threshold=0.2)

# # Predict labels for video segments
# labels = predict_video_labels(video_path)
# print(labels)

# # Class levels
# class_labels = ["Reach", "Pick", "Tilt", "Place", "Release", "Retract"]

# # Convert predicted integer labels to class names
# predicted_classes = []
# previous_prediction = None

# for label in labels:
#     class_name = class_labels[label]
#     if class_name != previous_prediction:
#         predicted_classes.append(class_name)
#         previous_prediction = class_name

# # Print the predicted class names
# print("Predicted Class Names:", predicted_classes)






#######################################################################################################################

'''Code to find the Frame Numbers with significant correlation changes in the video only one like 149 to 150 not from 150 to 151'''

# import os
# import cv2
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# import time

# # Load the trained model
# try:
#     model = load_model('Saved_models/New_subtasks_VGG16(26_3-06-2024).h5')
# except Exception as e:
#     print(f"Error loading model: {e}")
#     exit(1)

# def min_max_normalize(data): 
#     min_val = np.min(data)
#     max_val = np.max(data)
#     normalized_data = (data - min_val) / (max_val - min_val)
#     return normalized_data.reshape(224, 224, 3)

# # Function to extract frames from a video and calculate variance image
# def calculate_pixel_variance(frames):
#     num_frames = len(frames)
#     frame_height, frame_width = frames[0].shape[:2]

#     # Initialize an array to hold the variances
#     variance_array = np.zeros((frame_height, frame_width, 3))

#     # Iterate over each pixel position
#     for i in range(frame_height):
#         for j in range(frame_width):
#             # Extract the pixel values across all frames for the current pixel position
#             pixel_values = [frames[k][i, j] for k in range(num_frames)]

#             # Calculate the variance of the pixel values
#             variance_array[i, j] = np.var(pixel_values, axis=0)

#     return variance_array

# # Function to calculate the correlation between two frames
# def calculate_frame_correlation(frame1, frame2):
#     frame1_flat = frame1.flatten()
#     frame2_flat = frame2.flatten()
#     correlation = np.corrcoef(frame1_flat, frame2_flat)[0, 1]
#     return correlation

# # Function to compute and print correlation between consecutive frames in a video
# def print_frame_correlations(video_path, target_size=(224, 224), threshold=0.2):
#     cap = cv2.VideoCapture(video_path)
#     ret, prev_frame = cap.read()
#     if not ret:
#         print("Error reading video")
#         cap.release()
#         return
    
#     prev_frame = cv2.resize(prev_frame, target_size)
#     prev_frame = prev_frame / 255.0

#     frame_idx = 1
#     prev_correlation = None
#     significant_change_frames = []
#     significant_change_detected = False

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         frame = cv2.resize(frame, target_size)
#         frame = frame / 255.0
        
#         correlation = calculate_frame_correlation(prev_frame, frame)
#         print(f"Correlation between frame {frame_idx-1} and frame {frame_idx}: {correlation}")
        
#         if prev_correlation is not None:
#             abs_diff = abs(correlation - prev_correlation)
#             if abs_diff > threshold and not significant_change_detected:
#                 significant_change_frames.append(frame_idx)
#                 significant_change_detected = True
#             elif abs_diff <= threshold:
#                 significant_change_detected = False
        
#         prev_frame = frame
#         prev_correlation = correlation
#         frame_idx += 1

#     cap.release()
#     return significant_change_frames

# # Function to predict labels for a given video
# def predict_video_labels(video_path, segment_size=150, target_size=(224, 224)):
#     cap = cv2.VideoCapture(video_path)
#     frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     print(frame_count)
#     total_frames = frame_count
    
#     labels = []

#     for start_frame in range(0, total_frames, segment_size):
#         video_frames = []
#         for i in range(start_frame, start_frame + segment_size):
#             if i >= total_frames: 
#                 break
#             cap.set(cv2.CAP_PROP_POS_FRAMES, i)
#             ret, frame = cap.read()
#             if not ret:
#                 continue
#             frame = cv2.resize(frame, target_size)
#             frame = frame / 255.0
#             video_frames.append(frame)
        
#         if len(video_frames) <= segment_size:
#             variance_frame = calculate_pixel_variance(video_frames)
#             variance_frame = min_max_normalize(variance_frame)
#             variance_frame = np.expand_dims(variance_frame, axis=0)  # Normalize and add batch dimension
#             predicted_label = np.argmax(model.predict(variance_frame), axis=-1)
#             labels.append(predicted_label[0])
#     cap.release()
#     return labels

# def capture_and_save_video(video_path):
#     cap = cv2.VideoCapture(0)  # Use the default webcam (change to the appropriate index if you have multiple webcams)
#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

#     start_time = time.time()

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         elapsed_time = time.time() - start_time
#         minutes = int(elapsed_time // 60)
#         seconds = int(elapsed_time % 60)
#         timer_text = f'Time: {minutes:02}:{seconds:02}'

#         # Add the timer text to the frame
#         cv2.putText(frame, timer_text, (10, frame_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

#         out.write(frame)

#         # Display the captured frame
#         cv2.imshow('frame', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to stop capturing
#             break

#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()

# # Example usage
# video_path = 'Video_for_testing/27_6_merge_5.mp4'

# # To capture video and save it, you can call capture_and_save_video
# # capture_and_save_video(video_path)

# # Compute and print correlations between consecutive frames
# significant_change_frames = print_frame_correlations(video_path, threshold=0.2)
# print("Frames with significant correlation changes:", significant_change_frames)

# # Predict labels for video segments
# labels = predict_video_labels(video_path)
# print(labels)

# # Class levels
# class_labels = ["Reach", "Pick", "Tilt", "Place", "Release", "Retract"]

# # Convert predicted integer labels to class names
# predicted_classes = []
# previous_prediction = None

# for label in labels:
#     class_name = class_labels[label]
#     if class_name != previous_prediction:
#         predicted_classes.append(class_name)
#         previous_prediction = class_name

# # Print the predicted class names
# print("Predicted Class Names:", predicted_classes)






#####################################################################################################################

'''Code to find the Frame Numbers with significant correlation changes in the video 
then pass to the calculate the variance till significant change frame number and so on '''

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import time

# Load the trained model
try:
    model = load_model('C:/Users/Acer/Desktop/VGG16_variances/30 Frames Results/Var_30_frames_Vgg16_12_07.h5',compile=False)
    # model = load_model('C:/Users/Acer/Desktop/CVPR/Results_5_sec_subtask/Saved_models_5 seconds/New_subtasks_VGG16(26_3-06-2024).h5',compile=False)
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

def min_max_normalize(data): 
    min_val = np.min(data)
    max_val = np.max(data)
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data.reshape(224, 224, 3)

# Function to extract frames from a video and calculate variance image
def calculate_pixel_variance(frames):
    num_frames = len(frames)
    frame_height, frame_width = frames[0].shape[:2]

    # Initialize an array to hold the variances
    variance_array = np.zeros((frame_height, frame_width, 3))

    # Iterate over each pixel position
    for i in range(frame_height):
        for j in range(frame_width):
            # Extract the pixel values across all frames for the current pixel position
            pixel_values = [frames[k][i, j] for k in range(num_frames)]

            # Calculate the variance of the pixel values
            variance_array[i, j] = np.var(pixel_values, axis=0)

    return variance_array

# Function to calculate the correlation between two frames
def calculate_frame_correlation(frame1, frame2):
    frame1_flat = frame1.flatten()
    frame2_flat = frame2.flatten()
    correlation = np.corrcoef(frame1_flat, frame2_flat)[0, 1]
    return correlation

# Function to compute and print correlation between consecutive frames in a video
def get_significant_change_frames(video_path, target_size=(224, 224), threshold=0.2):
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    if not ret:
        print("Error reading video")
        cap.release()
        return []
    
    prev_frame = cv2.resize(prev_frame, target_size)
    prev_frame = prev_frame / 255.0

    frame_idx = 1
    prev_correlation = None
    significant_change_frames = []
    significant_change_detected = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, target_size)
        frame = frame / 255.0
        
        correlation = calculate_frame_correlation(prev_frame, frame)
        print(f"Correlation between frame {frame_idx-1} and frame {frame_idx}: {correlation}")
        
        if prev_correlation is not None:
            abs_diff = prev_correlation-correlation
            # abs_diff = abs(correlation - prev_correlation)
            if abs_diff > threshold and not significant_change_detected:
                significant_change_frames.append(frame_idx)
                significant_change_detected = True
            elif abs_diff <= threshold:
                significant_change_detected = False
        
        prev_frame = frame
        prev_correlation = correlation
        frame_idx += 1

    cap.release()
    return significant_change_frames


# Function to predict labels based on variance frames
def predict_labels_from_variance_frames(video_path, change_frames, target_size=(224, 224)):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    change_frames = [0] + change_frames + [total_frames]
    # print(len(change_frames))
    labels = []

    for i in range(len(change_frames) - 1):
        start_frame = change_frames[i]
        end_frame = change_frames[i + 1]

        video_frames = []
        for frame_idx in range(start_frame, end_frame):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, target_size)
            frame = frame / 255.0
            video_frames.append(frame)

        if video_frames:
            variance_frame = calculate_pixel_variance(video_frames)
            variance_frame = min_max_normalize(variance_frame)
            variance_frame = np.expand_dims(variance_frame, axis=0)  # Normalize and add batch dimension
            predicted_label = np.argmax(model.predict(variance_frame), axis=-1)
            labels.append(predicted_label[0])

    cap.release()
    return labels

def capture_and_save_video(video_path):
    cap = cv2.VideoCapture(0)  # Use the default webcam (change to the appropriate index if you have multiple webcams)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        elapsed_time = time.time() - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        timer_text = f'Time: {minutes:02}:{seconds:02}'

        # Add the timer text to the frame
        cv2.putText(frame, timer_text, (10, frame_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        out.write(frame)

        # Display the captured frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to stop capturing
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Example usage
video_path = 'Testing_videos/final_result1.mp4'
# video_path = 'Video_for_testing/realtime.mp4'

# To capture video and save it, you can call capture_and_save_video
# capture_and_save_video(video_path)

# Get frames where correlation changes significantly
significant_change_frames = get_significant_change_frames(video_path, threshold=0.08)
print("Frame Numbers with significant correlation changes:", significant_change_frames)

# Predict labels for segments between significant correlation changes
labels = predict_labels_from_variance_frames(video_path,significant_change_frames)
# labels = predict_labels_from_variance_frames(video_path,[120,210,300,390])
print(labels)

# Class levels
class_labels = ["Reach", "Pick", "Tilt", "Place", "Release", "Retract","Stir"]

# Convert predicted integer labels to class names
predicted_classes = [class_labels[label] for label in labels]

# Print the predicted class names
print("Predicted Class Names:", predicted_classes)
