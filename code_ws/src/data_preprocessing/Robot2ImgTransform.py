import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Ellipse
from scipy.spatial.transform import Rotation as R
import extract_data
import cv2
import odometry_transformations as ot
import csv

odoms = extract_data.extract_odom("/media/philip/Elements/capra_recordings/logs/20250205_1314/capra_robot_odometry.log") #List of odometry points from log, given as list
image_list = extract_data.extract_image_from_folder("/media/philip/Elements/capra_recordings/data/zed/20250205_1314/")
imu_data = extract_data.extract_imu_data("/media/philip/Elements/capra_recordings/data/imu/20250205_1314imu_data.csv")
ir_imgs_list = extract_data.extract_image_from_folder("/media/philip/Elements/capra_recordings/data/flir/20250205_1314/")

imu_data = extract_data.refine_imu(imu_data, 0.25)
image, ir_imgs = extract_data.correct_for_mismatch(ir_imgs_list,image_list)

output_path = '/media/philip/Elements/speciale_dataset/visualization/patches/'
#Exctract odometry points in the world frame (homogeneous coordinates)
odometry_points,orientations,timestamps, T_W_to_R = [], [], [], []
first_index = 0
for i in range(len(odoms['timestamp'])):
    if first_index == 0: #odoms['timestamp'][i] < odoms['timestamp'][len(odoms['timestamp'])-1] and 
        first_index = i
        position_in_R = np.array([odoms['pos_x'][i],odoms['pos_y'][i], odoms['pos_z'][i], 1])
        orientation_in_robot_frame = np.array([odoms['ori_x'][i], odoms['ori_y'][i], odoms['ori_z'][i], odoms['ori_w'][i]])
        timestamps.append(odoms['timestamp'][i])
        odometry_points.append(position_in_R.reshape(4,1))
        orientations.append(orientation_in_robot_frame.reshape(4,))

        R_mat = R.from_quat(orientation_in_robot_frame).as_matrix()
        T_mat = np.eye(4)

        T_mat[:3,:3] = R_mat
        T_mat[:3,3] = position_in_R[:3]

        T_W_to_R.append(T_mat)
    elif first_index != 0: # and odoms['timestamp'][i]-odoms['timestamp'][first_index]<=20 and odoms['timestamp'][i] < odoms['timestamp'][len(odoms['timestamp'])-1]:
        position_in_R = np.array([odoms['pos_x'][i],odoms['pos_y'][i], odoms['pos_z'][i], 1])
        orientation_in_robot_frame = np.array([odoms['ori_x'][i], odoms['ori_y'][i], odoms['ori_z'][i], odoms['ori_w'][i]])
        timestamps.append(odoms['timestamp'][i])
        odometry_points.append(position_in_R.reshape(4,1))
        orientations.append(orientation_in_robot_frame.reshape(4,))
        R_mat = R.from_quat(orientation_in_robot_frame).as_matrix()
        T_mat = np.eye(4)

        T_mat[:3,:3] = R_mat
        T_mat[:3,3] = position_in_R[:3]

        T_W_to_R.append(T_mat)
 

start_position = np.array([[odoms['pos_x'][first_index], odoms['pos_y'][first_index], odoms['pos_z'][first_index], 1]]).reshape(4,1)
start_orientation = T_W_to_R[0][:3,:3]

odometry_points = np.array(odometry_points)
orientations = np.array(orientations)

#print(odometry_points)
T_W_to_R = np.array(T_W_to_R)

# Corrected Translation vector [X (forward), Y (left), Z (up)]
translation = np.array([0.27, 0.06, 0.36])

# Euler angles in degrees (Roll, Pitch, Yaw)
euler_angles_deg =[0.30138765965066716 ,19.883949488337844 ,-0.12438729693569311]#[0.056310629792643736, 20.184583199425543, -1.1489344248906082]#

# Convert Euler angles to radians
euler_angles_rad = np.deg2rad(euler_angles_deg)

# Create rotation matrix (using XYZ -> Roll-Pitch-Yaw)
rotation_matrix = R.from_euler('xyz', euler_angles_rad).as_matrix()

# Construct the homogeneous transformation matrix (4x4)
T_R_to_C = np.eye(4)
T_R_to_C[:3, :3] = rotation_matrix  # Insert rotation
T_R_to_C[:3, 3] = translation       # Insert corrected translation


# Camera intrinsics
zed_intrinsics = np.array([[514.80341605, 0., 644.92250454],
                    [0. , 514.95300697, 348.70670139],
                    [0.,0.,1.]], dtype=np.float32)
flir_intrinsics = np.array([[755.74748203,0.,317.3601524],
                    [0.0,755.32090533,262.23083607],
                    [0., 0., 1.]], dtype=np.float32)
# World to camera transformation
T_C_to_I = np.array([[0,0,1,0],[-1,0,0,0],[0,-1,0,0],[0,0,0,1]])

# Transform between camera and IR camera obtained via calibration
Homography_matrix = np.array([
    [ 6.97681726e-01, -9.41330241e-02,  4.22102466e+02],
 [ 7.27601050e-03,  6.14641809e-01, 2.01762029e+02],
 [ 1.79244328e-05, -1.43605929e-04, 1.00000000e+00]])

# Calculate temporal characteristics 
elapsed_time = (odoms['timestamp'][len(odoms['timestamp'])-1]-odoms['timestamp'][first_index])%60
time_pr_image = elapsed_time/len(image)
imu_time = imu_data['timestamp'].max()-imu_data['timestamp'].min()
T_W_to_I_list = [] 
time_indencies = []


timestamps = np.array(timestamps)
for i in range(len(image)):

    time_index = np.abs(timestamps-timestamps[0]-i*time_pr_image).argmin()
    time_indencies.append(time_index)
    T_W_to_I_list.append(ot.transform_world_to_image(T_C_to_I, T_R_to_C, T_W_to_R[time_index]))
   
T_W_to_I_list = np.array(T_W_to_I_list)

image_width, image_height = 1280,720
flir_resolution = (640,512)
flir_cnrs = np.array([[0,0,1],
                      [flir_resolution[0]-1,0,1],
                      [flir_resolution[0]-1,flir_resolution[1]-1,1],
                      [0,flir_resolution[1]-1,1]])

odom_imu = ot.align_imu_and_odom(imu_data,odometry_points)

ideal_positions = ot.find_ideal_pos(start_position,start_orientation,20,10,1)
ideal_positions = np.array(ideal_positions)
scaled_ideal = ot.ideal_to_odom_stepsize(ideal_positions,odometry_points)
patch_trav = []

ot.make_output_dir(output_path)
number_of_patches = 6

for i in range(len(image)):


    image_points, valid_index = ot.project_robot_positions_to_image(
        odometry_points, T_W_to_I_list[i], zed_intrinsics, image_width, image_height
    )
    imgpts = np.array(image_points)
    valid_points, bounds, cropped_index = ot.crop_to_IR(imgpts,valid_index,Homography_matrix,flir_cnrs)
    rgb_segments = ot.segment_region_validation(bounds, number_of_patches, 2)
    ir_segments = ot.segment_region_validation(flir_cnrs[:,:2],number_of_patches,2)
    print(bounds)
    if i == 56:
        for k in range(len(rgb_segments)):
            rgb_dir = ot.crop_image_to_patch(rgb_segments[k],k,i,image[i],output_path)
            ir_dir = ot.crop_image_to_patch_IR(ir_segments[k],k,i,ir_imgs[i],output_path)#ot.crop_IR_to_patch(ir_imgs[i],i,k,number_of_patches,output_path)
    # #patches, patchcnrs = ot.divide_polygon_into_horizontal_patches(bounds, number_of_patches)
#     # for j in range(len(patchcnrs)):

#     #     resultimage = ot.draw_rectangle(image[i],patchcnrs[j],color=(0,255*patches[j]['confidence'],0)) 
#     # ot.plot_projected_points_on_image(resultimage,valid_points)
#     for k in range(len(patches)):
#         odoms_in_patch,pts_in_patch = ot.points_in_patch(valid_points,patchcnrs[k],cropped_index)
#         if odoms_in_patch.shape != (0,):
#             patch_trav.append(ot.calculate_patch_traversability(patches[k],odom_imu,odoms_in_patch,0.8,0.2,scaled_ideal,T_W_to_R, start_orientation))
#             rgb_dir = ot.crop_image_to_patch(patchcnrs[k],k,i,image[i],output_path)
#             ir_dir = ot.crop_IR_to_patch(ir_imgs[i],i,k,number_of_patches,output_path)



# rgb_files, ir_files = ot.get_file_names(rgb_dir,ir_dir)



# with open(output_path + 'annotations.csv', 'w', newline='') as csv_file:
#     headerList = ['RGB Image', 'IR image', 'Label1', 'Label2','Label 3', 'Confidence']
#     writer = csv.DictWriter(csv_file, fieldnames=headerList)
#     writer.writeheader()
#     for i,value in enumerate(patch_trav):
#         writer.writerow({'RGB Image': rgb_files[i], 'IR image': ir_files[i], 'Label1': value[0], 'Label2': value[1],'Label 3':value[2] ,'Confidence': value[3]})


    
      

# #print(len(T_W_to_R))
# # imgpts = np.array(image_points)

# print(len(scaled_ideal))
# ot.plot_xy_movement(odometry_points,scaled_ideal)







