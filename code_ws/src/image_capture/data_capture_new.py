import threading
import time
import pyzed.sl as sl
import cv2
import PySpin
import sys
import SaveToAvi




# Initialize a global flag for stopping threads
stop_event = threading.Event()

# Thread function for image recording and saving videos to avi 
def record_zed(zed, video_writer_zed):
    runtime = sl.RuntimeParameters()
    image = sl.Mat()
    
    
    images = list()
    try:
        while not stop_event.is_set():
            # Grab a frame from the ZED camera
            if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_image(image, sl.VIEW.LEFT)
                frame = image.get_data()
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                video_writer_zed.write(frame_bgr)

    except Exception as e:
        print(f"Error in record_video: {e}")

    finally:
        video_writer_zed.release()

def record_flir(flir_ax5, nodemap_tldevice, nodemap):
    # Set acquisition mode to continuous
    node_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
    if not PySpin.IsReadable(node_acquisition_mode) or not PySpin.IsWritable(node_acquisition_mode):
        print('Unable to set acquisition mode to continuous (enum retrieval). Aborting...')
        return False

    # Retrieve entry node from enumeration node
    node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
    if not PySpin.IsReadable(node_acquisition_mode_continuous):
        print('Unable to set acquisition mode to continuous (entry retrieval). Aborting...')
        return False
    acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()
    node_acquisition_mode.SetIntValue(acquisition_mode_continuous)
    print('Acquisition mode set to continuous...')

    # # Set pixel format to RGB
    # node_pixel_format = PySpin.CEnumerationPtr(nodemap.GetNode('PixelFormat'))
    # if not PySpin.IsAvailable(node_pixel_format) or not PySpin.IsWritable(node_pixel_format):
    #    print('Unable to set pixel format.. Aborting...')
    #    return False
    # node_pixel_format_rbg8 = PySpin.CEnumEntryPtr(node_pixel_format.GetEntryByName('RGB8'))
    # if not PySpin.IsAvailable(node_pixel_format_rbg8) or not PySpin.IsReadable(node_pixel_format_rbg8):
    #     print('Unable to set pixel format.. Aborting...')
    #     return False
    # pixel_format_rbg8 = node_pixel_format_rbg8.GetValue()
    # node_pixel_format.SetIntValue(pixel_format_rbg8)
    

    # Set the temperature resolution to high
    node_temp_linear = PySpin.CEnumerationPtr(nodemap.GetNode('TemperatureLinearResolution'))
    if not PySpin.IsAvailable(node_temp_linear) or not PySpin.IsWritable(node_temp_linear):
        print('Unable to set temperature resolution.. Aborting...')
        return False
    node_temp_linear_high = PySpin.CEnumEntryPtr(node_temp_linear.GetEntryByName('High'))
    if not PySpin.IsAvailable(node_temp_linear_high) or not PySpin.IsReadable(node_temp_linear_high):
        print('Unable to set temperature resolution.. Aborting...')
        return False
    linear_high = node_temp_linear_high.GetValue()
    node_temp_linear.SetIntValue(linear_high)    

    # Set the CMOS bit depth to x: 8, 14, 16
    node_bit_depth = PySpin.CEnumerationPtr(nodemap.GetNode('CMOSBitDepth'))
    if not PySpin.IsAvailable(node_bit_depth) or not PySpin.IsWritable(node_bit_depth):
        print('Unable to set CMOS bit depth.. Aborting...')
        return False
    node_bit_depth_xbit = PySpin.CEnumEntryPtr(node_bit_depth.GetEntryByName('bit8bit'))
    if not PySpin.IsAvailable(node_bit_depth_xbit) or not PySpin.IsReadable(node_bit_depth_xbit):
        print('Unable to set CMOS bit depth.. Aborting...')
        return False
    bit_depth = node_bit_depth_xbit.GetValue()
    node_bit_depth.SetIntValue(bit_depth)


    # Turn on temperature linear mode
    node_temp_linear = PySpin.CEnumerationPtr(nodemap.GetNode('TemperatureLinearMode'))
    if not PySpin.IsAvailable(node_temp_linear) or not PySpin.IsWritable(node_temp_linear):
        print('Unable to set temperature linear mode.. Aborting...')
        return False
    node_temp_linear_on = PySpin.CEnumEntryPtr(node_temp_linear.GetEntryByName('On'))
    if not PySpin.IsAvailable(node_temp_linear_on) or not PySpin.IsReadable(node_temp_linear_on):
        print('Unable to set temperature linear mode.. Aborting...')
        return False
    node_on = node_temp_linear_on.GetValue()
    node_temp_linear.SetIntValue(node_on)   
    
    #Start acquisition of IR images
    flir_ax5.BeginAcquisition()
    
    images = list()
    try:
        while not stop_event.is_set():
            _, image_result = SaveToAvi.acquire_images(flir_ax5)
            images.append(image_result)

    except Exception as e: 
        print(f"error in record_flir{e}")
    
    finally:
        result = SaveToAvi.save_list_to_avi(nodemap, nodemap_tldevice, images, framerate=7.49, filename='/media/jetson/Elements/capra_recordings/data/')

        #print(result)
        print(result)
        flir_ax5.EndAcquisition()
        flir_ax5.DeInit()

def record_imu(zed, imu_log_file):
    sensors_data = sl.SensorsData()
    
    try:
        while not stop_event.is_set():
            # Retrieve sensors data
            if zed.get_sensors_data(sensors_data, sl.TIME_REFERENCE.CURRENT) == sl.ERROR_CODE.SUCCESS:
                imu_data = sensors_data.get_imu_data()
                acceleration = imu_data.get_linear_acceleration()
                angular_velocity = imu_data.get_angular_velocity()
                
                imu_log_file.write(f"{time.time()}, {acceleration[0]}, {acceleration[1]}, {acceleration[2]}, "
                                   f"{angular_velocity[0]}, {angular_velocity[1]}, {angular_velocity[2]}\n")

    except Exception as e:
        print(f"Error in record_imu: {e}")

    finally:
        # Ensure the log file is closed
        imu_log_file.close()

def wait_for_stop():
    print("Recording... Press Enter to stop.")
    input()  # Changed from sys.stdin.read(1) to input()
    stop_event.set()

def main():
    global stop_event
    
    # Initialize ZED camera
    zed = sl.Camera()
    init_params = sl.InitParameters(depth_mode=sl.DEPTH_MODE.NEURAL)
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 15
    
    # Flir initialization
    device_id = '73500261'  # taken from SpinView
    system = PySpin.System.GetInstance()
    camera_list = system.GetCameras()
    flir_ax5 = camera_list.GetBySerial(device_id)
    nodemap_tldevice = flir_ax5.GetTLDeviceNodeMap()
    
    flir_ax5.Init()
    #flir_ax5.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
    
    nodemap = flir_ax5.GetNodeMap()
   

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Failed to open ZED camera.")
        return

    # Setup OpenCV VideoWriter for saving video
    #image_size = zed.get_camera_information().camera_configuration.resolution
    image_size_width = zed.get_camera_information().camera_resolution.width
    image_size_height = zed.get_camera_information().camera_resolution.height
    video_writer_zed = cv2.VideoWriter('/media/jetson/Elements/capra_recordings/data/zed_output_new.avi', cv2.VideoWriter_fourcc(*'MJPG'), 15,
                                       (image_size_width, image_size_height))
    

    # Open file for IMU logging
    imu_log_file = open("/media/jetson/Elements/capra_recordings/data/imu_data.csv", "w")
    imu_log_file.write("timestamp, accel_x, accel_y, accel_z, ang_vel_x, ang_vel_y, ang_vel_z\n")

    # Start threads
    zed_thread = threading.Thread(target=record_zed, args=(zed, video_writer_zed))
    flir_thread = threading.Thread(target=record_flir, args= (flir_ax5, nodemap_tldevice, nodemap))
    imu_thread = threading.Thread(target=record_imu, args=(zed, imu_log_file))
    stop_thread = threading.Thread(target=wait_for_stop)
    
    # Set stop_thread as a daemon thread
    stop_thread.daemon = True
    
    zed_thread.start()
    flir_thread.start()
    imu_thread.start()
    stop_thread.start()

    # Wait for the stop thread to complete
    stop_thread.join()

    # Wait for the other threads to finish
    flir_thread.join()
    zed_thread.join()
    imu_thread.join()
    
    # Cleanup
    video_writer_zed.release()
    imu_log_file.close()
    zed.close()
    del flir_ax5
    camera_list.Clear()
    system.ReleaseInstance()

if __name__ == "__main__":
    main()
