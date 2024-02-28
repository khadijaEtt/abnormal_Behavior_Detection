from ultralytics import YOLO
import shutil
import tensorflow as tf
import cv2
import numpy as np

def extract_frames(video_path):
    frames = []
    frame_width = 224
    frame_height = 224
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_per_batch = 10
    skip_frames = max(total_frames // frames_per_batch, 1)  # Ensure at least one frame per batch

    i = 0
    while True:
        batch_frames = []
        for j in range(frames_per_batch):
            frame_index = i * skip_frames + j
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (frame_width, frame_height))
                batch_frames.append(frame)
            else:
                break  # Break if unable to read frame

        if not batch_frames:
            break  # Break if no more frames are available

        frames.append(np.array(batch_frames))
        i += 1

    cap.release()
    return frames

def import_file(source_path, destination_path):
    try:
        # Copy the file from source_path to destination_path
        shutil.copy(source_path, destination_path)
        print(f"File successfully imported from {source_path} to {destination_path}")
    except FileNotFoundError:
        print("File not found. Check the source path.")
    except PermissionError:
        print("Permission error. Check your file permissions.")
    except Exception as e:
        print(f"An error occurred: {e}")

def load_model():
  model=tf.keras.models.load_model('./final_model (2).h5')
  return model

# Example usage:
source_file_path = 'C:/Users/Lenovo/runs/pose/predict/hugging_vid.avi'
destination_folder_path = 'C:/Users/Lenovo/Desktop/prjTransv/result_yolov8'

#import_file(source_file_path, destination_folder_path)
'''
# Load a model
model = YOLO('yolov8n-pose.pt')  # load an official model
# Predict with the model
results = model(source='hugging_vid.mp4', show=False, conf=0.3, save=True)  # predict on an image
print(results)
'''
frames_per_video = 10
TestVideo = 'C:/Users/Lenovo/runs/pose/predict2/hugging_vid.avi'
model = load_model()
input_frames = extract_frames(TestVideo)

for frames in input_frames:
        frames = np.expand_dims(frames, axis=0)
        predictions = model.predict(frames)
        print(predictions)
        predicted_class = np.argmax(predictions, axis=1)

        # Print the predicted class
        print(f'Predicted class: {predicted_class[0]}')


