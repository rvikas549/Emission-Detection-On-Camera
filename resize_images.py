# resize_images.py

import cv2
import os

# Source and destination directories
input_root = 'dataset/raw_screenshots'
output_root = 'dataset/resized'
target_size = (128, 128)

# Create output folders
for class_name in ['no_smoke', 'light_smoke', 'heavy_smoke']:
    input_dir = os.path.join(input_root, class_name)
    output_dir = os.path.join(output_root, class_name)
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(input_dir, filename)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Skipped unreadable image: {filename}")
                continue
            resized = cv2.resize(img, target_size)
            save_path = os.path.join(output_dir, filename)
            cv2.imwrite(save_path, resized)
            print(f"Saved: {save_path}")
