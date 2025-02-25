import paho.mqtt.client as mqtt
import logging
import datetime 
# MQTT broker settings
BROKER = "10.46.28.1"
PORT = 1883
TOPICS = ["/capra/robot/odometry", "/capra/robot/geo_odometry", "/capra/robot/status/core"]  # Replace with your desired topics

logs_folder = datetime.datetime.now().strftime("%Y%m%d_%H%M")
# Log file settings
LOG_DIR = "/media/jetson/Elements/capra_recordings/logs/"+logs_folder+'/'  # Directory to store log files for each topic

# Ensure the log directory exists
import os
os.makedirs(LOG_DIR, exist_ok=True)

# Callback when the client connects to the broker
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected successfully")
        for topic in TOPICS:
            client.subscribe(topic)
            print(f"Subscribed to topic: {topic}")
    else:
        print(f"Failed to connect with code {rc}")

# Callback when a message is received from the broker
def on_message(client, userdata, msg):
    message = msg.payload.decode("utf-8")
    topic = msg.topic
    log_file = os.path.join(LOG_DIR, f"{topic.replace('/', '_')}.log")
    
    # Write to the topic-specific log file
    with open(log_file, "a") as f:
        f.write(f"{message}\n")
    
    print(f"Logged message from {topic}: {message}")

# Initialize MQTT client
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

try:
    print(f"Connecting to MQTT broker at {BROKER}:{PORT}")
    client.connect(BROKER, PORT, 60)

    # Blocking loop to process network traffic and dispatch callbacks
    client.loop_forever()
except KeyboardInterrupt:
    print("Exiting...")
    client.disconnect()
except Exception as e:
    print(f"An error occurred: {e}")
