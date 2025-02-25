# import threading
# import time
# import pyzed.sl as sl
# import cv2
# import PySpin
# import sys
# import SaveToAvi




# # Initialize a global flag for stopping threads
# stop_event = threading.Event()

# # Thread function for image recording and saving videos to avi 
# def record_video(zed, video_writer_zed, flir_ax5, nodemap_tldevice, nodemap):
#     runtime = sl.RuntimeParameters()
#     image = sl.Mat()
        
#     # Set acquisition mode to continuous
#     node_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
#     if not PySpin.IsReadable(node_acquisition_mode) or not PySpin.IsWritable(node_acquisition_mode):
#         print('Unable to set acquisition mode to continuous (enum retrieval). Aborting...')
#         return False

#     # Retrieve entry node from enumeration node
#     node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
#     if not PySpin.IsReadable(node_acquisition_mode_continuous):
#         print('Unable to set acquisition mode to continuous (entry retrieval). Aborting...')
#         return False

#     acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()

#     node_acquisition_mode.SetIntValue(acquisition_mode_continuous)

#     print('Acquisition mode set to continuous...')
    
#     #Start acquisition of IR images
#     flir_ax5.BeginAcquisition()
    
#     images = list()
#     try:
#         while not stop_event.is_set():
#             # Grab a frame from the ZED camera
#             if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
#                 zed.retrieve_image(image, sl.VIEW.LEFT)
#                 frame = image.get_data()
#                 frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
#                 video_writer_zed.write(frame_bgr)

#             # Grab a frame from the FLIR camera and adds it to the list 
#             _, image_result = SaveToAvi.acquire_images(flir_ax5)
#             images.append(image_result)
            
            
        
        

#     except Exception as e:
#         print(f"Error in record_video: {e}")

#     finally:
#         # Ensure FLIR acquisition stops and writers are released 
        
#         #flir_ax5.DeInit()
#         # Use updated SaveToAvi function to write images at 30 Hz to Path DATA
#         result = SaveToAvi.save_list_to_avi(nodemap, nodemap_tldevice, images, 15, '../Data/')

#         #print(result)
#         print(result)
#         flir_ax5.EndAcquisition()
#         flir_ax5.DeInit()
#         video_writer_zed.release()


# def record_imu(zed, imu_log_file):
#     sensors_data = sl.SensorsData()
    
#     try:
#         while not stop_event.is_set():
#             # Retrieve sensors data
#             if zed.get_sensors_data(sensors_data, sl.TIME_REFERENCE.CURRENT) == sl.ERROR_CODE.SUCCESS:
#                 imu_data = sensors_data.get_imu_data()
#                 acceleration = imu_data.get_linear_acceleration()
#                 angular_velocity = imu_data.get_angular_velocity()
                
#                 imu_log_file.write(f"{time.time()}, {acceleration[0]}, {acceleration[1]}, {acceleration[2]}, "
#                                    f"{angular_velocity[0]}, {angular_velocity[1]}, {angular_velocity[2]}\n")

#     except Exception as e:
#         print(f"Error in record_imu: {e}")

#     finally:
#         # Ensure the log file is closed
#         imu_log_file.close()

# def wait_for_stop():
#     print("Recording... Press Enter to stop.")
#     input()  # Changed from sys.stdin.read(1) to input()
#     stop_event.set()

# def main():
#     global stop_event
    
#     # Initialize ZED camera
#     zed = sl.Camera()
#     init_params = sl.InitParameters(depth_mode=sl.DEPTH_MODE.NEURAL)
#     init_params.camera_resolution = sl.RESOLUTION.HD1080
#     init_params.camera_fps = 15
    
#     # Flir initialization
#     device_id = '73500261'  # taken from SpinView
#     system = PySpin.System.GetInstance()
#     camera_list = system.GetCameras()
#     flir_ax5 = camera_list.GetBySerial(device_id)
#     nodemap_tldevice = flir_ax5.GetTLDeviceNodeMap()
    
#     flir_ax5.Init()
#     #flir_ax5.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
    
#     nodemap = flir_ax5.GetNodeMap()
   

#     if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
#         print("Failed to open ZED camera.")
#         return

#     # Setup OpenCV VideoWriter for saving video
#     image_size = zed.get_camera_information().camera_configuration.resolution
#     video_writer_zed = cv2.VideoWriter('../Data/zed_output.avi', cv2.VideoWriter_fourcc(*'MJPG'), 15,
#                                        (image_size.width, image_size.height))
    

#     # Open file for IMU logging
#     imu_log_file = open("../Data/imu_data.csv", "w")
#     imu_log_file.write("timestamp, accel_x, accel_y, accel_z, ang_vel_x, ang_vel_y, ang_vel_z\n")

#     # Start threads
#     video_thread = threading.Thread(target=record_video, args=(zed, video_writer_zed, 
#                                                                flir_ax5, nodemap_tldevice, nodemap))
#     imu_thread = threading.Thread(target=record_imu, args=(zed, imu_log_file))
#     stop_thread = threading.Thread(target=wait_for_stop)
    
#     # Set stop_thread as a daemon thread
#     stop_thread.daemon = True
    
#     video_thread.start()
#     imu_thread.start()
#     stop_thread.start()

#     # Wait for the stop thread to complete
#     stop_thread.join()

#     # Wait for the other threads to finish
#     video_thread.join()
#     imu_thread.join()
    
#     # Cleanup
#     video_writer_zed.release()
#     imu_log_file.close()
#     zed.close()
#     del flir_ax5
#     camera_list.Clear()
#     system.ReleaseInstance()

# if __name__ == "__main__":
#     main()

import csv
import numpy as np

csv_path = 'src/camera_calibration/input/csv/0_annotations.csv'

contours = []
with open(csv_path,'r') as csvfile:
    csvreader = csv.reader(csvfile)

    for row in csvreader:
        contours.append((int(row[1]),int(row[2])))
contour_list = np.array(contours,dtype=np.int32)

print(contour_list)
