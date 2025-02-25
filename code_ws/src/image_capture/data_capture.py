import threading
import time
import pyzed.sl as sl
import cv2
import PySpin
import os
import datetime
import paho.mqtt.client as mqtt
import json
import SaveToAvi
import numpy as np

# Initialize a global flag for stopping threads
stop_event = threading.Event()

# MQTT broker configuration
BROKER_ADDRESS = "10.46.28.1"  # Change to your broker's IP or hostname
BROKER_PORT = 1883            # Default MQTT port
TOPIC = 'capra/remote/direct_velocity'  # Topic to publish velocity commands

# MQTT client setup
client = mqtt.Client()

def on_message(client, userdata, message):
    try:
        payload = message.payload.decode('utf-8')
        data = json.loads(payload)
        linear_x = data['twist']['linear']['x']
        steering_angle = data['twist']['angular']['z']
        
        if linear_x == 0 and steering_angle == 0:
            print("Received stop command.")
            stop_event.set()
    except Exception as e:
        print(f"Error processing MQTT message: {e}")

def setup_mqtt():
    client.on_message = on_message
    client.connect(BROKER_ADDRESS, BROKER_PORT, 60)
    client.subscribe(TOPIC)
    client.loop_start()

# Thread function for image recording and saving videos to avi 
def record_zed(zed):
    runtime = sl.RuntimeParameters()
    image = sl.Mat()
    images = list()
    try:
        while not stop_event.is_set():
            if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_image(image, sl.VIEW.LEFT)
                frame = image.get_data()
                frame_copy = np.copy(frame)
                images.append(frame_copy)
    except Exception as e:
        print(f"Error in record_video: {e}")
    finally:
        frame_id = 0
        folder_name = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        os.makedirs('/media/jetson/Elements/capra_recordings/data/zed/'+folder_name, exist_ok=True)
        for frame in images:
            filename = '/media/jetson/Elements/capra_recordings/data/zed/'+str(folder_name)+'/'+str(frame_id)+'.jpg'
            cv2.imwrite(filename, frame)
            frame_id += 1

def record_flir(flir_ax5, nodemap_tldevice, nodemap):
    # Setup FLIR camera as before
    # Start acquisition of IR images
    flir_ax5.BeginAcquisition()
    images = list()
    try:
        while not stop_event.is_set():
            _, image_result = SaveToAvi.acquire_images(flir_ax5)
            images.append(image_result)
    except Exception as e: 
        print(f"error in record_flir{e}")
    finally:
        processor = PySpin.ImageProcessor()
        frame_id = 0
        folder_name = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        os.makedirs('/media/jetson/Elements/capra_recordings/data/flir/'+folder_name, exist_ok=True)
        for frame in images:
            converted_img = processor.Convert(frame, PySpin.PixelFormat_Mono8)
            filename = '/media/jetson/Elements/capra_recordings/data/flir/'+str(folder_name)+'/'+str(frame_id)+'.jpg'
            converted_img.Save(filename)
            frame_id += 1
        flir_ax5.EndAcquisition()
        flir_ax5.DeInit()

def record_imu(zed, imu_log_file):
    sensors_data = sl.SensorsData()
    try:
        while not stop_event.is_set():
            if zed.get_sensors_data(sensors_data, sl.TIME_REFERENCE.CURRENT) == sl.ERROR_CODE.SUCCESS:
                imu_data = sensors_data.get_imu_data()
                acceleration = imu_data.get_linear_acceleration()
                angular_velocity = imu_data.get_angular_velocity()
                imu_log_file.write(f"{time.time()}, {acceleration[0]}, {acceleration[1]}, {acceleration[2]}, "
                                   f"{angular_velocity[0]}, {angular_velocity[1]}, {angular_velocity[2]}\n")
    except Exception as e:
        print(f"Error in record_imu: {e}")
    finally:
        imu_log_file.close()

def main():
    global stop_event
    setup_mqtt()

    # Initialize ZED camera
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 15

    # Flir initialization
    device_id = '73500261'  # taken from SpinView
    system = PySpin.System.GetInstance()
    camera_list = system.GetCameras()
    flir_ax5 = camera_list.GetBySerial(device_id)
    nodemap_tldevice = flir_ax5.GetTLDeviceNodeMap()
    flir_ax5.Init()
    nodemap = flir_ax5.GetNodeMap()

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Failed to open ZED camera.")
        return

    imu_id = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    imu_log_file = open("/media/jetson/Elements/capra_recordings/data/imu/"+imu_id+"imu_data.csv", "w")
    imu_log_file.write("timestamp, accel_x, accel_y, accel_z, ang_vel_x, ang_vel_y, ang_vel_z\n")

    # Start threads
    zed_thread = threading.Thread(target=record_zed, args=(zed,))
    flir_thread = threading.Thread(target=record_flir, args=(flir_ax5, nodemap_tldevice, nodemap))
    imu_thread = threading.Thread(target=record_imu, args=(zed, imu_log_file))

    zed_thread.start()
    flir_thread.start()
    imu_thread.start()

    # Wait for the stop event to be set
    stop_event.wait()

    # Wait for the other threads to finish
    flir_thread.join()
    zed_thread.join()
    imu_thread.join()

    # Cleanup
    imu_log_file.close()
    zed.close()
    del flir_ax5
    camera_list.Clear()
    system.ReleaseInstance()
    client.loop_stop()
    client.disconnect()

if __name__ == "__main__":
    main()