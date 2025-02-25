import time
import json
import paho.mqtt.client as mqtt

# MQTT broker configuration
BROKER_ADDRESS = "10.46.28.1"  # Change to your broker's IP or hostname
BROKER_PORT = 1883            # Default MQTT port
TOPIC = 'capra/remote/direct_velocity'  # Topic to publish velocity commands

# Initialize MQTT client
client = mqtt.Client()
client.connect(BROKER_ADDRESS, BROKER_PORT)

def publish_velocity(linear_x, angular_z):
    message = {
        "header": {
            "frame_id": "frame_id"
        },
        "twist": {
            "linear": {
                "x": linear_x
            },
            "angular": {
                "z": angular_z
            }
        }
    }
    client.publish(TOPIC, json.dumps(message))

# Start publishing at 10 Hz for 5 seconds
publish_rate_hz = 10
publish_interval = 1.0 / publish_rate_hz
start_time = time.time()

try:
    while time.time() - start_time < 3:
        publish_velocity(1.0, 0.0)
        time.sleep(publish_interval)
    
    # Send stop command after 5 seconds
    publish_velocity(0.0, 0.0)
    print("Published stop command.")
except KeyboardInterrupt:
    publish_velocity(0.0, 0.0)
    print("Interrupted. Published stop command.")

client.disconnect()
