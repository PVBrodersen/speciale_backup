import pandas as pd
import matplotlib.pyplot as plt
import extract_data
# Load the CSV file
file_path = '/media/philip/Elements/capra_recordings/data/20241217_1614imu_data.csv'
raw_df = extract_data.extract_imu_data(file_path)
df = extract_data.refine_imu(raw_df,0.25)#pd.read_csv(file_path)#extract_data.extract_imu_data(file_path)

df.columns = df.columns.str.strip()
print(df.dtypes)
# Convert timestamp to seconds relative to the first timestamp
df['timestamp'] = df['timestamp'] - df['timestamp'].iloc[0]
df = df[df['timestamp']<=20.5]

# Create a figure with two subplots (Acceleration & Angular Velocity)
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

# Plot acceleration
axes[0].plot(df['timestamp'].values, df['accel_x'].values, label='Accel X', color='r')
axes[0].plot(df['timestamp'].values, df['accel_y'].values, label='Accel Y', color='g')
axes[0].plot(df['timestamp'].values, df['accel_z'].values, label='Accel Z', color='b')
axes[0].set_ylabel("Acceleration (m/s²)")
axes[0].legend()
axes[0].grid()

# Plot angular velocity
axes[1].plot(df['timestamp'].values, df['ang_vel_x'].values, label='Ang Vel X', color='r')
axes[1].plot(df['timestamp'].values, df['ang_vel_y'].values, label='Ang Vel Y', color='g')
axes[1].plot(df['timestamp'].values, df['ang_vel_z'].values, label='Ang Vel Z', color='b')
axes[1].set_ylabel("Angular Velocity (rad/s)")
axes[1].set_xlabel("Time (seconds)")
axes[1].legend()
axes[1].grid()

plt.suptitle("IMU Data Over Time")
plt.show()