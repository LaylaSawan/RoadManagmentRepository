import boto3
import cv2
import time
import os
import json
import numpy as np
from ultralytics import YOLO

# AWS Credentials
AWS_S3_BUCKET_NAME = 'gettings3bucketimages'
AWS_REGION = 'us-east-2'
AWS_ACCESS_KEY = 'AKIATSXH5I6FM3N4UUGL'
AWS_SECRET_KEY = ''

# Load YOLOv8 model trained for pothole detection
model = YOLO('yolov8n-seg.pt')

def get_unique_filename(base_name="video"):
    return f"{base_name}_{int(time.time())}.mp4"

# Initialize webcam
webcam = cv2.VideoCapture(0)
frame_width, frame_height = int(webcam.get(3)), int(webcam.get(4))
fps = 20
video_writer, recording = None, False
video_filename = ""

def process_video(video_filename):
    print("Processing video to detect potholes...")
    cap = cv2.VideoCapture(video_filename)
    damage_data = []
    output_filename = video_filename.replace(".mp4", "_processed.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, imgsz=640, conf=0.25, classes=[0])  # Ensure it detects potholes only
        
        total_area = 0
        if results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()
            image_area = frame.shape[0] * frame.shape[1]
            for mask in masks:
                binary_mask = (mask > 0).astype(np.uint8) * 255
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    total_area += cv2.contourArea(contours[0])
                    cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)  # Draw green outlines for potholes

        damage_percentage = (total_area / image_area) * 100 if total_area > 0 else 0.0
        damage_data.append({
            "frame": int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
            "damage_percentage": float(damage_percentage)
        })
        out.write(frame)

    cap.release()
    out.release()
    json_filename = video_filename.replace(".mp4", "_damage.json")
    with open(json_filename, "w") as f:
        json.dump(damage_data, f, indent=4)
    print(f"Saved damage data: {json_filename}")
    upload_to_s3(output_filename, json_filename)

def upload_to_s3(video_filename, json_filename):
    s3_client = boto3.client(
        's3', region_name=AWS_REGION, aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY)
    
    s3_client.upload_file(video_filename, AWS_S3_BUCKET_NAME, video_filename)
    s3_client.upload_file(json_filename, AWS_S3_BUCKET_NAME, json_filename)
    print("Uploaded to S3!")
    
    signed_url = s3_client.generate_presigned_url('get_object', Params={'Bucket': AWS_S3_BUCKET_NAME, 'Key': video_filename}, ExpiresIn=3600)
    print(f"Signed URL: {signed_url}")
    os.remove(video_filename)
    os.remove(json_filename)
    print("Local files deleted.")

while True:
    print("Press 'r' to record, 's' to stop/save, 'q' to quit.")
    check, frame = webcam.read()
    cv2.imshow("Recording Window", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('r') and not recording:
        video_filename = get_unique_filename()
        video_writer = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
        recording = True
        print(f"Recording: {video_filename}")

    elif key == ord('s') and recording:
        recording = False
        video_writer.release()
        print(f"Saved: {video_filename}")
        process_video(video_filename)

    elif key == ord('q'):
        if recording:
            video_writer.release()
        webcam.release()
        cv2.destroyAllWindows()
        print("Exited.")
        break
    
    if recording:
        video_writer.write(frame)
