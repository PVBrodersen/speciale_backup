import re
import json
import pandas as pd
import matplotlib.pyplot as plt

# Path to the log file
log_file_path = '/media/philip/Elements/capra_recordings/logs/20241213_1310/capra_robot_odometry.log'

# Define the regex pattern to locate `twist` data
pattern = re.compile(
    r'"twist":\{"twist":\{"linear":\{"x":(?P<linear_x>[-\d.]+),"y":(?P<linear_y>[-\d.]+),"z":(?P<linear_z>[-\d.]+)\},'
    r'"angular":\{"x":(?P<angular_x>[-\d.]+),"y":(?P<angular_y>[-\d.]+),"z":(?P<angular_z>[-\d.]+)\}\}'
)

# Initialize lists to store parsed data
data = {
    "timestamp": [],
    "linear_x": [],
    "linear_y": [],
    "linear_z": [],
    "angular_x": [],
    "angular_y": [],
    "angular_z": [],
}

# Read and parse the log file
with open(log_file_path, 'r') as file:
    for line in file:
        match = pattern.search(line)
        if match:
            # Parse the twist data
            twist_data = match.groupdict()
            data["linear_x"].append(float(twist_data["linear_x"]))
            data["linear_y"].append(float(twist_data["linear_y"]))
            data["linear_z"].append(float(twist_data["linear_z"]))
            data["angular_x"].append(float(twist_data["angular_x"]))
            data["angular_y"].append(float(twist_data["angular_y"]))
            data["angular_z"].append(float(twist_data["angular_z"]))
            
            # Optionally, extract timestamp if available in the line
            timestamp_match = re.search(r'"seconds":(\d+),"nanoseconds":(\d+)', line)
            if timestamp_match:
                seconds = int(timestamp_match.group(1))
                nanoseconds = int(timestamp_match.group(2))
                timestamp = seconds + nanoseconds * 1e-9
                data["timestamp"].append(timestamp)

# Check if data was extracted successfully
print(f"Extracted {len(data['timestamp'])} entries of twist data.")

# Convert to a DataFrame for easier handling
twist_df = pd.DataFrame(data)

# Display the first few rows
print(twist_df.head())

# Plot the data side-by-side
fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharex=True)

# Plot linear components
axes[0].plot(twist_df["timestamp"], twist_df["linear_x"], label="Linear X")
axes[0].plot(twist_df["timestamp"], twist_df["linear_y"], label="Linear Y")
axes[0].plot(twist_df["timestamp"], twist_df["linear_z"], label="Linear Z")
axes[0].set_title("Linear Velocity Components Over Time")
axes[0].set_xlabel("Time (s)")
axes[0].set_ylabel("Linear Velocity (m/s)")
axes[0].legend()
axes[0].grid()

# Plot angular components
axes[1].plot(twist_df["timestamp"], twist_df["angular_x"], label="Angular X")
axes[1].plot(twist_df["timestamp"], twist_df["angular_y"], label="Angular Y")
axes[1].plot(twist_df["timestamp"], twist_df["angular_z"], label="Angular Z")
axes[1].set_title("Angular Velocity Components Over Time")
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("Angular Velocity (rad/s)")
axes[1].legend()
axes[1].grid()

# Show the plots
plt.tight_layout()
plt.show()
