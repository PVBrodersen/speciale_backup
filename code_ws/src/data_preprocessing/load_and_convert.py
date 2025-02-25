import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# Function to load images from a folder as NumPy arrays
def load_images_from_folder(folder_path):
    image_list = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            with Image.open(file_path) as img:
                img_in_celcius = np.asarray(img)*0.04
                image_list.append(np.asarray(img_in_celcius))

        except Exception as e:
            print(f"Could not load image {file_path}: {e}")
    return image_list

# Path to the folder containing images
folder_path = "/media/philip/Elements/capra_recordings/data/20241213_1311"  # Replace with your folder path
image_list = load_images_from_folder(folder_path)
output_name = "heatmapped"
output_path = os.path.join(folder_path,output_name)
os.makedirs(output_path, exist_ok=True)

if not image_list:
    raise ValueError("The image list is empty. Add some images to display.")

current_index = 0

# Function to update the displayed image
def update_image():
    ax.clear()
    ax.imshow(image_list[current_index], cmap='inferno', aspect='auto')
    ax.set_title(f"Image {current_index + 1}/{len(image_list)} Max temp: {np.round(np.max(image_list[current_index]), decimals= 2)} Min temp: {np.round(np.min(image_list[current_index]), decimals= 2)}")
    ax.axis('off')
    fig.canvas.draw()

# Function for saving
def save_current_image(output_folder, image):
    output_img = output_folder+'/'+image
    temp_fig , temp_ax = plt.subplots()
    temp_ax.imshow(image_list[current_index], cmap='inferno', aspect='auto')
    plt.axis('off')
    plt.savefig(output_img, bbox_inches='tight', pad_inches=0)
    plt.close(temp_fig)
    print(f"Image saved to {output_folder}")

# Key event handler
def on_key(event):
    global current_index

    if event.key == 'right':  # Go to the next image
        if current_index < len(image_list) - 1:
            current_index += 1
            update_image()
    elif event.key == 'left':  # Go to the previous image
        if current_index > 0:
            current_index -= 1
            update_image()
    elif event.key == 'down':
        save_current_image(output_path,f"saved_image_{current_index+1}.png")    
    elif event.key == 'escape':  # Exit on Esc key
        plt.close(fig)

# Create a Matplotlib figure and axes
fig, ax = plt.subplots()
update_image()

# Connect the key event to the handler
fig.canvas.mpl_connect('key_press_event', on_key)

# Show the plot
plt.show()
