import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Ellipse
from scipy.spatial.transform import Rotation as R
import cv2
import os
from shapely.geometry import Polygon, Point
def transform_world_to_camera(P_W, T_W_to_C):
    """
    Transforms a point from the world frame to the camera frame.

    Parameters:
    - P_W: 4x1 homogeneous point in world frame.
    - T_W_to_R: 4x4 transformation matrix from world to robot frame.
    - T_R_to_C: 4x4 transformation matrix from robot to camera frame.
    
    Returns:
    - P_C: 4x1 homogeneous point in the camera frame.
    """

    #flip_x = np.array([1,1,1,1])    
    P_C = T_W_to_C @ P_W
    P_C = P_C #* (flip_x.reshape(4,1))
    return P_C

def project_to_image(P_C, K, image_width, image_height):
    """Projects a 3D point in the camera frame onto the 2D image plane."""
    x_c, y_c, z_c, _ = P_C.flatten()
    
    # Skip points behind the camera
    if z_c <= 0:
        return None

    # Perspective projection
    p_img = K @ np.array([x_c, y_c, z_c])
    u = p_img[0] / p_img[2]
    v = p_img[1] / p_img[2]

    # Check if the point is inside the image
    if 0 <= u < image_width and 0 <= v < image_height:
        return (int(u), int(v))
    else:
        return None

def project_robot_positions_to_image(future_positions, T_W_to_C, K, image_width, image_height):
    """Projects future robot positions into the camera image."""
    image_points = []
    index = []
    for i, P_W in enumerate(future_positions):
        #T_W_to_R = T_W_to_R_list[i]
        P_C = transform_world_to_camera(P_W, T_W_to_C)
        
        pixel_coords = project_to_image(P_C, K, image_width, image_height)
        
        if pixel_coords is not None:
            image_points.append(pixel_coords)
            index.append(i)

    return image_points, index

def plot_projected_points_on_image(image, image_points):
    """
    Loads an image and plots projected points on it.

    Parameters:
    - image_path: Path to the image file.
    - image_points: List of (u, v) pixel coordinates to plot.
    """
    # Load the image
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for plotting

    # Plot projected points with gradually decreasing circle size
    total_points = len(image_points)
    for i, (u, v) in enumerate(image_points):
        # Smooth decrease in radius, starting from 10 and reducing to 3
        radius = int(25 - (7 * (i / total_points)))
        radius = max(3, radius)  # Ensure the radius doesn't go below 3
        cv2.circle(image_rgb, (u, v), radius=radius, color=(255, 0, 0), thickness=-1)  # Filled red dots

    # Display the image with points
    plt.figure(figsize=(10, 8))
    plt.imshow(image_rgb)
    plt.title("Projected Robot Positions on Image")
    plt.axis("off")
    plt.show()    
def plot_xy_movement(odometry_points_1, odometry_points_2):
    """
    Plots the XY movement of two sets of odometry points on the same plot.
    
    Parameters:
    - odometry_points_1: numpy array of shape (N, 4), first set of odometry points [x, y, z, 1]
    - odometry_points_2: numpy array of shape (M, 4), second set of odometry points [x, y, z, 1]
    """
    # Extract X and Y coordinates for the first set
    x_coords_1 = odometry_points_1[:, 0]
    y_coords_1 = odometry_points_1[:, 1]

    # Extract X and Y coordinates for the second set
    x_coords_2 = odometry_points_2[:, 0]
    y_coords_2 = odometry_points_2[:, 1]

    # Plot both trajectories
    plt.figure(figsize=(8, 6))
    plt.plot(x_coords_1, y_coords_1, marker='o', linestyle='-', color='blue', label="Actual Trajectory")
    plt.plot(x_coords_2, y_coords_2, marker='o', linestyle='-', color='red', label="Ideal Trajectory")
    
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("XY Movement Trajectories")
    plt.grid(True)
    plt.legend()
    plt.axis("equal")  # Ensures equal scaling for x and y axes
    plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()
    plt.show()
    
# def find_ideal_pos(start_pos, time, frequency, linear_velocity, steering_angle):
#     ideal_positions = []

#     for i in range(time*frequency):
#         temp_pos = np.array([start_pos[0][0]+np.sin(steering_angle)*linear_velocity*0.1*i,
#         start_pos[1][0]-np.cos(steering_angle)*linear_velocity*0.1*i,
#         start_pos[2][0],
#         1
#         ])
#         ideal_positions.append(temp_pos.reshape(4,1))

#     return ideal_positions   

def find_ideal_pos(start_pos, start_orientation, time, frequency, linear_velocity):
    ideal_positions = []

    # Define the local motion direction (forward in the robot's x-direction)
    local_direction = np.array([1, 0, 0])  # [x, y, z] in the robot's frame

    # Transform the local direction into the world frame using the rotational matrix
    world_direction = start_orientation @ local_direction  # R * local_direction

    for i in range(time * frequency):
        # Calculate displacement based on the world direction and linear velocity
        displacement = linear_velocity * 0.1 * i

        # Calculate the new position
        temp_pos = np.array([
            start_pos[0][0] + world_direction[0] * displacement,
            start_pos[1][0] + world_direction[1] * displacement,
            start_pos[2][0],  # Z remains unchanged (assuming 2D motion)
            1  # Homogeneous coordinate
        ])

        # Append the new position to the list
        ideal_positions.append(temp_pos.reshape(4, 1))

    return ideal_positions

def transform_world_to_image(T_C_to_I, T_R_to_C, T_W_to_R):
    T_W_to_I = np.linalg.inv(T_C_to_I) @np.linalg.inv(T_R_to_C) @ np.linalg.inv(T_W_to_R)
    return T_W_to_I

def crop_to_IR(imagepoints, index_vec, homography, ir_corners):

    corners_in_rgb = []
    valid_img_pts = []
    cropped_index = []
    for pts in ir_corners:
        temp = np.dot(homography,pts)
        corners_in_rgb.append(temp[:2]/temp[2])

    bounds = np.array(corners_in_rgb)
    straight_bounds = straighten_ROI(bounds)
    imgpts = np.array(imagepoints)
    for i,imgpt in enumerate(imgpts):
        if(straight_bounds[:,0].min()<imgpt[0]<straight_bounds[:,0].max() and straight_bounds[:,1].min()<imgpt[1]<straight_bounds[:,1].max()):
            valid_img_pts.append(imgpt)
            cropped_index.append(index_vec[i])
        else:
            continue

    return np.array(valid_img_pts), straight_bounds, cropped_index

def sort_corners_clockwise(corners, reference=(0, 0)):
    """
    Sorts corners in clockwise order starting from the top-left corner.
    Args:
        corners: List of (x, y) tuples representing the corners.
        reference: (x, y) reference point for the top-left corner. Default is (0, 0).
    Returns:
        Sorted corners in clockwise order starting from the closest to the reference.
    """
    # Compute the centroid of the polygon
    centroid = np.mean(corners, axis=0)
    
    # Calculate angles relative to the centroid
    angles = np.arctan2(corners[:, 1] - centroid[1], corners[:, 0] - centroid[0])
    
    # Sort corners by angle in clockwise order
    sorted_indices = np.argsort(angles)
    sorted_corners = corners[sorted_indices]
    
    # Find the corner closest to the reference (0, 0) and rotate the sorted list
    distances_to_reference = np.linalg.norm(sorted_corners - reference, axis=1)
    start_index = np.argmin(distances_to_reference)
    sorted_corners = np.roll(sorted_corners, -start_index, axis=0)
    
    return sorted_corners
def interpolate_edges(corners, y):
    """
    Interpolates x-coordinates along the edges of the polygon for a given y.
    Returns the x-range as [x_min, x_max].
    """
    edges = [
        (corners[i], corners[(i + 1) % len(corners)])  # Wrap around to form edges
        for i in range(len(corners))
    ]
    
    x_coords = []
    for p1, p2 in edges:
        y1, y2 = p1[1], p2[1]
        if y1 == y2:  # Horizontal edge
            if y == y1:  # If y matches this horizontal edge
                x_coords.append(p1[0])
                x_coords.append(p2[0])
        elif y1 <= y <= y2 or y2 <= y <= y1:  # Check if y lies within the edge's y-range
            # Linear interpolation for x
            x = p1[0] + (p2[0] - p1[0]) * (y - y1) / (y2 - y1)
            x_coords.append(x)
    
    return [min(x_coords), max(x_coords)] if x_coords else [None, None]
def divide_polygon_into_horizontal_patches(corners, n):
    """
    Divides a polygon into n horizontal patches along the y-axis.
    Args:
        corners: List of 4 (x, y) tuples representing the polygon corners.
        n: Number of patches.
    Returns:
        List of patches with their coordinates and confidence values.
    """
    # Output arrays
    patch_cnrs = []
    # Ensure corners are sorted clockwise
    corners = np.array(corners)
    sorted_corners = sort_corners_clockwise(corners)
    
    # Extract y-range and divide into patches
    y_min, y_max = np.min(sorted_corners[:, 1]), np.max(sorted_corners[:, 1])
    y_boundaries = np.linspace(y_min, y_max, n + 1)

    patches = []
    for i in range(n):
        # Current y-boundaries
        y_low = y_boundaries[i]
        y_high = y_boundaries[i + 1]
        
        # Interpolate along edges to find x-boundaries at y_low and y_high
        if(i==0):
            x_low = [corners[0,0],corners[1,0]]
        else:
            x_low = interpolate_edges(sorted_corners, y_low)
        if(i!=n-1):    
            x_high = interpolate_edges(sorted_corners, y_high)
        else:
            x_high = [corners[3,0],corners[2,0]]
        
        # Compute patch center and confidence
        y_center = (y_low + y_high) / 2
        confidence = (y_center - y_min) / (y_max - y_min)
        
        # Create a patch with confidence
        patches.append({
            "x_low": x_low,
            "x_high": x_high,
            "y_min": y_low,
            "y_max": y_high,
            "confidence": confidence
        })

        # Patch corners
        patch_cnrs.append([[x_low[0],y_low],[x_low[1],y_low],[x_high[1],y_high],[x_high[0],y_high]])

    return patches, np.array(patch_cnrs)
def draw_rectangle(image, points, color=(0, 255, 0), thickness=2):
    """
    Draws a rectangle on the image using the provided corner points.

    Parameters:
        image (numpy.ndarray): The input image where the rectangle will be drawn.
        points (list of tuples): List of 4 pixel coordinates (x, y) as corner points.
                                The points should be in the following order:
                                top-left, top-right, bottom-right, bottom-left.
        color (tuple): Color of the rectangle (B, G, R).
        thickness (int): Thickness of the rectangle edges.

    Returns:
        numpy.ndarray: The image with the rectangle drawn.
    """
    if len(points) != 4:
        raise ValueError("Exactly 4 points are required to draw a rectangle.")

    # Convert points to an integer numpy array
    points = np.array(points, dtype=np.int32)

    # Draw lines connecting the points to form a rectangle
    for i in range(4):
        start_point = tuple(points[i])
        end_point = tuple(points[(i + 1) % 4])  # Wrap around to the first point
        cv2.line(image, start_point, end_point, color, thickness)

    return image
def align_imu_and_odom(imu_data, odom_data):
    imu_pr_odom = int(len(imu_data)/len(odom_data))

    odom_imu_weight = []
    for i in range(len(odom_data)):

        if i == 0: 
            accel_x_avg = imu_data['accel_x'][i:i+imu_pr_odom].mean()
            accel_y_avg = imu_data['accel_y'][i:i+imu_pr_odom].mean()
            accel_z_avg = imu_data['accel_z'][i:i+imu_pr_odom].mean()
            ang_vel_x_avg = imu_data['ang_vel_x'][i:i+imu_pr_odom].mean()
            ang_vel_y_avg = imu_data['ang_vel_y'][i:i+imu_pr_odom].mean()
            ang_vel_z_avg = imu_data['ang_vel_z'][i:i+imu_pr_odom].mean()
        if i == len(odom_data) - 1:
            accel_x_avg = imu_data['accel_x'][i*imu_pr_odom:].mean()
            accel_y_avg = imu_data['accel_y'][i*imu_pr_odom:].mean()
            accel_z_avg = imu_data['accel_z'][i*imu_pr_odom:].mean()
            ang_vel_x_avg = imu_data['ang_vel_x'][i*imu_pr_odom:].mean()
            ang_vel_y_avg = imu_data['ang_vel_y'][i*imu_pr_odom:].mean()
            ang_vel_z_avg = imu_data['ang_vel_z'][i*imu_pr_odom:].mean()
        else:
            accel_x_avg = imu_data['accel_x'][i*imu_pr_odom:(i*imu_pr_odom)+imu_pr_odom].mean()
            accel_y_avg = imu_data['accel_y'][i*imu_pr_odom:(i*imu_pr_odom)+imu_pr_odom].mean()
            accel_z_avg = imu_data['accel_z'][i*imu_pr_odom:(i*imu_pr_odom)+imu_pr_odom].mean()
            ang_vel_x_avg = imu_data['ang_vel_x'][i*imu_pr_odom:(i*imu_pr_odom)+imu_pr_odom].mean()
            ang_vel_y_avg = imu_data['ang_vel_y'][i*imu_pr_odom:(i*imu_pr_odom)+imu_pr_odom].mean()
            ang_vel_z_avg = imu_data['ang_vel_z'][i*imu_pr_odom:(i*imu_pr_odom)+imu_pr_odom].mean()

        odom_imu_weight.append([[accel_x_avg,accel_y_avg,accel_z_avg],[ang_vel_x_avg,ang_vel_y_avg,ang_vel_z_avg]])

    return odom_imu_weight
def straighten_ROI(ROI_cnrs):
    
    if(ROI_cnrs[0][1]>ROI_cnrs[1][1]):
        top_y = ROI_cnrs[0][1]
    else:
        top_y = ROI_cnrs[1][1]
    if(ROI_cnrs[2][1]<ROI_cnrs[3][1]):
        bot_y = ROI_cnrs[2][1]
    else:
        bot_y = ROI_cnrs[3][1]
    if(ROI_cnrs[0][0]<ROI_cnrs[3][0]):
        left_x = ROI_cnrs[3][0]
    else:
        left_x = ROI_cnrs[0][0]
    if(ROI_cnrs[1][0]<ROI_cnrs[2][0]):
        right_x = ROI_cnrs[1][0]
    else:
        right_x = ROI_cnrs[2][0]
    straight_cnr = [[left_x,top_y],[right_x,top_y],[right_x,bot_y],[left_x,bot_y]]

    return np.array(straight_cnr)          
def ideal_to_odom_stepsize(ideal_pos, odometries):

    length_ideal_path_x = ideal_pos[len(ideal_pos)-1,0]-ideal_pos[0,0]
    length_ideal_path_y = ideal_pos[len(ideal_pos)-1,1]-ideal_pos[0,1]

    scaled_ideal = []
    for i in range(len(odometries)):
        scaled_x = ideal_pos[0,0] + i*(length_ideal_path_x/len(odometries))
        scaled_y = ideal_pos[0,1] + i*(length_ideal_path_y/len(odometries))

        scaled_ideal.append((scaled_x,scaled_y,ideal_pos[0,2]))
    return np.array(scaled_ideal)



def calculate_patch_traversability(patch, odom_imu_weights, odom_index, acc_weight, vel_weight, ideal_positions, T_W_to_R, start_orientation):
    odom_info = []
    

    for odom in odom_index:
        acc_norm = np.linalg.norm(odom_imu_weights[odom][0])
        vel_norm = np.linalg.norm(odom_imu_weights[odom][1])
        combined_imu = acc_weight*acc_norm+vel_weight*vel_norm
        
        delta_pos = np.linalg.norm(T_W_to_R[odom][:3,3]-ideal_positions[odom].T)
        relative_rotation = np.dot(start_orientation.T,T_W_to_R[odom][:3,:3])
        trace = np.trace(relative_rotation)
        theta = np.arccos(np.clip((trace - 1) / 2, -1.0, 1.0))
        
        sign = np.sign(relative_rotation[1, 0] - relative_rotation[0, 1])  # From skew-symmetric part

        signed_theta = sign * theta

        odom_info.append([delta_pos,signed_theta,combined_imu])


    odom_info=np.array(odom_info)
    if odom_info.shape != (0,):
        patch_trav_deviation_ang = np.sum(odom_info[:,1])
        patch_trav_deviation_pos = np.sum(odom_info[:,0])
        patch_trav_impact = np.sum(odom_info[:,2])
        confidence = patch['confidence']
        #print('Traversability for patch',patch['confidence'],':',patch_trav)

        
    return patch_trav_deviation_ang, patch_trav_deviation_pos, patch_trav_impact, confidence

def inverse_sigmoid(input, a=0.01, b=200):
    out = 1/(1+np.exp(a*(input-b)))
    return out


def points_in_patch(cropped_points, patch_cnrs, cropped_index):

    pts_in_patch = []
    ir_points = np.array(cropped_points)
    bound = np.array(patch_cnrs)
    cropped_odom_index = []

    for i, pts in enumerate(ir_points):
        if bound[:, 0].min() <= pts[0] <= bound[:, 0].max() and bound[:, 1].min() <= pts[1] <= bound[:, 1].max():
            pts_in_patch.append(pts)
            cropped_odom_index.append(cropped_index[i])
    #print(len(pts_in_patch))
    return np.array(cropped_odom_index), np.array(pts_in_patch)


def crop_image_to_patch(patch_cnrs, patch_nr, image_nr, image, output_path):

    img = image

    output_folder = output_path+'RGB_patches/'

    if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    cropped_img = img[int(patch_cnrs[0,1]):int(patch_cnrs[2,1]),int(patch_cnrs[0,0]):int(patch_cnrs[2,0])]
    if cropped_img is not None and cropped_img.size > 0:
        cv2.imwrite(output_folder+'image_'+str(image_nr)+'_patch_'+str(patch_nr)+'.jpg',cropped_img)
    else:
        print("Invalid image data")
    

    return output_folder     
def crop_image_to_patch_IR(patch_cnrs, patch_nr, image_nr, image, output_path):

    img = image

    output_folder = output_path+'IR_patches/'

    if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    cropped_img = img[int(patch_cnrs[0,1]):int(patch_cnrs[2,1]),int(patch_cnrs[0,0]):int(patch_cnrs[2,0])]
    if cropped_img is not None and cropped_img.size > 0:
        cv2.imwrite(output_folder+'image_'+str(image_nr)+'_patch_'+str(patch_nr)+'.jpg',cropped_img)
    else:
        print("Invalid image data")
    

    return output_folder 


def crop_IR_to_patch(image, image_nr, patch_nr, number_of_patches, output_path):

    img = image
    height, width = img.shape[:2]
    patch_heigt = int(height/number_of_patches)

    output_folder = output_path+'IR_patches/'
    
    if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    cropped_img = img[patch_nr*patch_heigt:(patch_nr+1)*patch_heigt, 0:width]

    if cropped_img is not None and cropped_img.size > 0:
        cv2.imwrite(output_folder+'image_'+str(image_nr)+'_patch_'+str(patch_nr)+'.jpg',cropped_img)
    else:
        print("Invalid image data")

    return output_folder

def get_file_names(RGB_dir, IR_dir):

    rgb_files = os.listdir(RGB_dir)
    ir_files = os.listdir(IR_dir)

    return rgb_files,ir_files

def make_output_dir (output_path):

    os.makedirs(output_path, exist_ok=True)


def segment_region_validation(corners, N, center_spacing):
    """
    Segments a quadrilateral region into fixed-width, vertically spaced segments.
    
    Parameters:
        corners (array-like): Four (x, y) points defining the region.
        N (int): Number of equal-height segments along the total height.
        center_spacing (float): Vertical spacing (dy) between segment centers.
    
    Returns:
        List of valid segment corners, each as a tuple of 4 corner points.
    """
    # Convert to NumPy array for easy indexing
    corners = np.array(corners)
    
    # Define the polygon region
    polygon = Polygon(corners)

    # Get bounding box of the region
    min_x, min_y, max_x, max_y = polygon.bounds

    # Compute segment width (equal to polygon width)
    segment_w = max_x - min_x

    # Compute segment height (total height divided into N parts)
    segment_h = (max_y - min_y) / N

    # Compute fixed x-position (center of the bounding box width)
    x_center = (min_x + max_x) / 2  

    # List to store valid segment corners
    valid_segments = []

    # Iterate over potential segment centers along the y-axis
    y = min_y + segment_h / 2  # Start at half-segment height
    while y + segment_h / 2 <= max_y:  # Ensure full segment stays inside the region
        center = Point(x_center, y)
        if polygon.contains(center):  # Ensure segment center is inside the region
            # Compute segment corners
            top_left = (x_center - segment_w / 2, y - segment_h / 2)
            top_right = (x_center + segment_w / 2, y - segment_h / 2)
            bottom_left = (x_center - segment_w / 2, y + segment_h / 2)
            bottom_right = (x_center + segment_w / 2, y + segment_h / 2)
            valid_segments.append((top_left, top_right, bottom_right, bottom_left))
        
        y += center_spacing  # Move to the next position

    return np.array(valid_segments)