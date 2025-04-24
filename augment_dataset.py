import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, array_to_img
from tqdm import tqdm

# Configuration
input_dir = "dataset/resized"
output_dir = "augmented_dataset"
target_images_per_class = 1000
img_size = (128, 128)  # Should match your model's input size

# Data augmentation generator
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    shear_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Create output directories if not exist
os.makedirs(output_dir, exist_ok=True)

# Loop through each class
for class_name in os.listdir(input_dir):
    class_input_path = os.path.join(input_dir, class_name)
    class_output_path = os.path.join(output_dir, class_name)
    os.makedirs(class_output_path, exist_ok=True)

    print(f"\n📂 Processing class: {class_name}")

    # Load original images
    images = [img for img in os.listdir(class_input_path) if img.lower().endswith(('.jpg', '.png', '.jpeg'))]
    num_existing = len(images)
    print(f"Found {num_existing} base images.")

    saved = 0
    loop_count = 0

    while saved < target_images_per_class:
        image_name = images[loop_count % num_existing]
        image_path = os.path.join(class_input_path, image_name)
        img = load_img(image_path, target_size=img_size)
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)

        # Generate batches of augmented images
        for batch in datagen.flow(x, batch_size=1):
            save_path = os.path.join(class_output_path, f"{class_name}_{saved + 1}.jpg")
            array_to_img(batch[0]).save(save_path)
            saved += 1
            if saved >= target_images_per_class:
                break
        loop_count += 1

    print(f"✅ Saved {saved} augmented images for class '{class_name}'")
