import time
import json
import paho.mqtt.client as mqtt
import math
import datetime
import csv

# MQTT broker configuration
BROKER_ADDRESS = "10.46.28.1"  # Change to your broker's IP or hostname
BROKER_PORT = 1883            # Default MQTT port
TOPIC = 'capra/remote/direct_velocity'  # Topic to publish velocity commands

# Initialize MQTT client
client = mqtt.Client()
client.connect(BROKER_ADDRESS, BROKER_PORT)

# Constants for circular motion
# linear_velocity = 1.0  # m/s
# radius = 1.32  # meters
wheelbase = 0.6  # meters, adjust to your robot's wheelbase
# Note: ved linear operation køres der med 1m/s i 3 sekunder. Den første meter er der markat pr. 20. cm og derefter en hver halve meter frem til 2m.
# Note forsat: ved cirkulær kørsel er der sat tape hver 45 grader. Det noteres også at robotten skidder lidt.  
# Calculate steering angle (angular_z) in radians
#steering_angle = math.atan(wheelbase / radius)

# Calculate time to complete one full circle
#time_to_complete_circle = (2 * math.pi * radius) / linear_velocity

# File path to save published messages
# log_file = datetime.datetime.now().strftime("%Y%m%d_%H%M")
# LOG_FILE_PATH = "/media/jetson/Elements/capra_recordings/logs/"+log_file+"-published_messages.txt"

def publish_velocity(linear_x, steering_angle):
    message = {
        "header": {
            "frame_id": "frame_id"
        },
        "twist": {
            "linear": {
                "x": linear_x
            },
            "angular": {
                "z": steering_angle
            }
        }
    }
    #message_json = json.dumps(message)
    client.publish(TOPIC, json.dumps(message))
    
    # # Save the published message to a text file
    # with open(LOG_FILE_PATH, "a", newline="") as csv_file:
    #     csv_writer = csv.writer(csv_file)
    #     csv_writer.writerow([time.time(), linear_x, steering_angle])

def oscillating_trajectory(duration, frequency, max_steering_angle):
    """
    Executes an oscillating trajectory by alternating the steering angle.
    
    :param duration: Total time to run the oscillation (seconds)
    :param frequency: Frequency of oscillation (Hz)
    :param max_steering_angle: Maximum steering angle in radians
    """
    start_time = time.time()
    period = 1.0 / frequency
    try:
        while time.time() - start_time < duration:
            elapsed_time = time.time() - start_time
            # Calculate oscillating steering angle using sine wave
            steering = max_steering_angle * math.sin(2 * math.pi * period * elapsed_time)
            publish_velocity(linear_velocity, steering)
            time.sleep(publish_interval)
        
        # Send stop command after oscillation
        publish_velocity(0.0, 0.0)
        print("Published stop command after oscillation.")
    except KeyboardInterrupt:
        publish_velocity(0.0, 0.0)
        print("Interrupted. Published stop command.")

# Start publishing at 10 Hz for the time to complete a full circle
publish_rate_hz = 10
publish_interval = 1.0 / publish_rate_hz
#start_time = time.time()

mode = input("Choose traversal mode (linear, angular, or oscillating): ")
if mode == 'angular':
    radius = float(input("Provide circle radius:"))
    steering_angle = math.atan(wheelbase / radius)
    linear_velocity = 1.0 
    time_to_complete_circle = (2 * math.pi * radius) / linear_velocity
    start_time = time.time()
    try:
        while time.time() - start_time < time_to_complete_circle:
            publish_velocity(linear_velocity, steering_angle)
            time.sleep(publish_interval)
        
        # Send stop command after completing the circle
        publish_velocity(0.0, 0.0)
        print("Published stop command.")
    except KeyboardInterrupt:
        publish_velocity(0.0, 0.0)
        print("Interrupted. Published stop command.")
elif mode == 'linear':
    duration = float(input("Enter duration for linear travel in seconds:"))
    start_time = time.time()
    try:
        while time.time() - start_time < duration:
            publish_velocity(1, 0)
            time.sleep(publish_interval)
        
        # Send stop command after completing the motion
        publish_velocity(0.0, 0.0)
        print("Published stop command.")
    except KeyboardInterrupt:
        publish_velocity(0.0, 0.0)
        print("Interrupted. Published stop command.")
elif mode == 'oscillating':
    duration = float(input("Enter duration of oscillation in seconds: "))
    frequency = float(input("Enter frequency of oscillation in Hz: "))
    max_angle = float(input("Enter maximum steering angle in radians: "))
    start_time = time.time()
    oscillating_trajectory(duration, frequency, max_angle)

client.disconnect()
