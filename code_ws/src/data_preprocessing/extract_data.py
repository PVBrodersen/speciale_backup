import numpy as np 
import csv
import re
import json
import cv2
import pandas as pd
import os 

def extract_odom(filepath):
    # Updated regex pattern to include 'w' in orientation
    pattern = re.compile(
    r'"pose":\s*\{"pose":\s*\{"position":\s*\{"x":\s*(?P<pos_x>[-\de.+]+),\s*"y":\s*(?P<pos_y>[-\de.+]+),\s*"z":\s*(?P<pos_z>[-\de.+]+)\},'
    r'\s*"orientation":\s*\{"x":\s*(?P<ori_x>[-\de.+]+),\s*"y":\s*(?P<ori_y>[-\de.+]+),\s*"z":\s*(?P<ori_z>[-\de.+]+),\s*"w":\s*(?P<ori_w>[-\de.+]+)\}'
    )
    odoms = {
        "timestamp": [],
        "pos_x": [],
        "pos_y": [],
        "pos_z": [],
        "ori_x": [],
        "ori_y": [],
        "ori_z": [],
        "ori_w": []
    }

    with open(filepath, 'r') as file:
        for line in file:
            # Match the pose pattern
            match = pattern.search(line)
            if match:
                odom_data = match.groupdict()
                odoms["pos_x"].append(float(odom_data["pos_x"]))
                odoms["pos_y"].append(float(odom_data["pos_y"]))
                odoms["pos_z"].append(float(odom_data["pos_z"]))
                odoms["ori_x"].append(float(odom_data["ori_x"]))
                odoms["ori_y"].append(float(odom_data["ori_y"]))
                odoms["ori_z"].append(float(odom_data["ori_z"]))
                odoms["ori_w"].append(float(odom_data["ori_w"]))

                # Extract timestamp from the same line if present
                timestamp_match = re.search(r'"seconds":(\d+),"nanoseconds":(\d+)', line)
                if timestamp_match:
                    seconds = int(timestamp_match.group(1))
                    nanoseconds = int(timestamp_match.group(2))
                    timestamp = seconds + nanoseconds * 1e-9
                    odoms["timestamp"].append(timestamp)
            else:
                print("No match found in line:", line.strip())

    return odoms

def extract_frames(video_path):
    """
    Extracts frames from a video file and returns them as an array of NumPy images.

    Args:
        video_path (str): Path to the video file.

    Returns:
        list: A list of frames as NumPy arrays (in BGR format).
    """
    frames = []
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if the video file is successfully opened
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video file: {video_path}")
    
    while True:
        # Read the next frame from the video
        ret, frame = cap.read()
        
        # If no frame is returned, the video has ended
        if not ret:
            break
        
        # Append the frame to the list
        frames.append(frame)
    
    # Release the video capture object
    cap.release()
    
    return frames

def extract_imu_data(csv_path):
    """
    Extracts IMU data from a CSV file and returns it as a DataFrame.

    Parameters:
    - csv_path (str): Path to the CSV file containing IMU data.

    Returns:
    - DataFrame: Cleaned DataFrame with IMU data.
    """
    # Load the CSV file
    df = pd.read_csv(csv_path)
    
    # Strip leading/trailing spaces from column names
    df.columns = df.columns.str.strip()
    
    # Extract relevant IMU data
    imu_data = df[['timestamp', 'accel_x', 'accel_y', 'accel_z', 
                   'ang_vel_x', 'ang_vel_y', 'ang_vel_z']]
    
    return imu_data


def refine_imu(df, threshold):
    """
    Extracts data points from a DataFrame after a significant derivative change in accel_x.

    Parameters:
        df (pd.DataFrame): Input DataFrame with columns ['timestamp', 'accel_x', 'accel_y', 'accel_z',
                                                         'ang_vel_x', 'ang_vel_y', 'ang_vel_z'].
        threshold (float): Threshold for the derivative change in accel_x.

    Returns:
        pd.DataFrame: Filtered DataFrame with rows after the significant derivative change.
    """
    # Calculate the derivative (difference) of accel_x
    df['accel_x_derivative'] = df['accel_x'].diff()
    
    # Find the first index where the absolute derivative exceeds the threshold
    change_index = df.index[(df['accel_x_derivative'].abs() > threshold)].min()
    
    # If no change is detected, return an empty DataFrame
    if pd.isna(change_index):
        return pd.DataFrame(columns=df.columns)
    
    # Return the DataFrame starting from the index after the detected change
    return df.loc[change_index + 1:].reset_index(drop=True)

def extract_image_from_folder(folder_path):

    images = [] 

    for filename in os.listdir(folder_path):

        file_path = os.path.join(folder_path, filename)

        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            # Read the image and append to the list
            img = cv2.imread(file_path)
            if img is not None:
                images.append(img)
    
    image_list = images

    return image_list

def correct_for_mismatch(IR_images, RGB_images):
    corrected_IR = []
    corrected_RGB = []
    if len(IR_images)>len(RGB_images):
        mismatch = int(len(IR_images)/len(RGB_images))
        for i in range(len(RGB_images)):
            corrected_IR.append(IR_images[i*mismatch])
        corrected_RGB = RGB_images
    else:
        mismatch = int(len(RGB_images)/len(IR_images))
        for i in range(len(IR_images)):
            corrected_RGB.append(RGB_images[i*mismatch])
        corrected_IR = IR_images
    

    return corrected_RGB, corrected_IR    